"""语音模块：离线唤醒词听指令，臂每走一步就说一句。

    python voice/wake.py                 # 听麦克风（必须在 MacBook 的「终端」App 里启动，SSH 起的拿不到麦克风）
    python voice/wake.py --wav x.wav     # 拿录音文件测识别，不动臂
    python voice/wake.py --dry           # 听麦克风，只打印，不调臂服务
    python voice/wake.py --chat-wav q.wav --dry   # 拿一句录音测自由对话（听写 → 判断命令/聊天 → 小思的声音），不调臂服务

听：sherpa-onnx 的关键词检测（KWS），不联网，不要账号。词表 voice/keywords_raw.txt，
    改完要重新生成 voice/keywords.txt（命令见 voice/README.md）。
说：macOS 自带 say（中文音色 Eddy）。先用它跑通，之后换别的 TTS 只改 speak()。
臂：只调臂服务 8000 端口（arm_service/server.py），不碰 5050、不碰串口。

词 → 动作在 ACTIONS；臂的每个 phase → 台词在 config/lines.json。

自由对话（有 MiMo 密钥才开：环境变量 XIAOSI_MIMO_KEY 或 ~/.xiaosi_mimo_key，见 brain/chat.py）：
    叫醒（「小思小思」）后接着说一句话，最多 6 秒 → 联网听写 → 是命令（停下/拿杯子/握手/确定…）就照口令走，
    要动臂的照样先问、等「小思确定」；不是命令就用小思的人格回一句，页面上出字幕。聊天本身从不动臂。
    没密钥、断网：跟以前一样只认口令。--no-chat 强制关掉。
"""

import argparse
import collections
import io
import json
import os
import pathlib
import queue
import subprocess
import sys
import threading
import time
import urllib.request
import wave

import numpy as np
import sherpa_onnx

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from brain import narrator  # noqa: E402  句子→接口、phase→台词都在大脑里查表
from brain import chat  # noqa: E402  自由对话（MiMo 听写 + 聊天 + 说话）

CHAT = False                                        # run_mic / --chat-wav 里按密钥打开
WAKE_WORDS = {narrator.NAME * 2, narrator.NAME + "同学"}

MODEL = ROOT / "models" / "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
KEYWORDS = ROOT / "voice" / "keywords.txt"
ARM_URL = os.environ.get("ARM_URL", "http://127.0.0.1:8000")
VOICE = os.environ.get("SAY_VOICE", "Eddy (中文（中国大陆）)")
SAY_RATE = os.environ.get("SAY_RATE", "150")      # 字/分钟，比默认慢，照顾老人；SAY_RATE=180 恢复正常

_say_q = queue.Queue()
_speaking = threading.Event()                       # 正在放声音：这时候不录用户说话（免得录到自己）
_play = {"end": 0.0, "n": 0}                          # 上一段声音放完的时刻、放完了几段（换了就重置识别）
SELF_GUARD_S = 1.0                                  # 小思说完 1 秒内听到的口令也不算（回声）
STOP_WORD = narrator.NAME + "停下"


def accept_keyword(word, now=None):
    """小思自己的台词里有「…然后说：小思确定」「小思松手」「叫我小思小思」，实测离线识别会听成口令。
    所以小思说话时、说完 1 秒内，只认「小思停下」，别的口令都丢掉，免得它自己确认、自己松手。"""
    now = time.time() if now is None else now
    talking = _speaking.is_set() or now - _play["end"] < SELF_GUARD_S
    return word == STOP_WORD or not talking


CLIPS = ROOT / "voice" / "clips"          # 小思的真人感语音：提前用小米 MiMo 生成好，断网也能放


def _clip_for(text):
    try:
        idx = json.loads((CLIPS / "index.json").read_text(encoding="utf-8"))
    except Exception:
        return None
    f = idx.get(text)
    return CLIPS / f if f and (CLIPS / f).exists() else None


def _say_worker():
    while True:
        text = _say_q.get()
        _speaking.set()
        clip = _clip_for(text) or (chat.tts_file(text) if CHAT else None)   # 没现成的就现场用 MiMo 生成
        try:
            if clip:                                       # 有现成的小思语音就放它
                subprocess.run(["afplay", str(clip)], check=False)
            else:                                          # 没有（比如带出错原因的句子）退回系统声音
                subprocess.run(["say", "-v", VOICE, "-r", SAY_RATE, text], check=False)
        except FileNotFoundError:
            print("[说不出来，没有 afplay/say]", text)
        _play["end"], _play["n"] = time.time(), _play["n"] + 1
        if _say_q.empty():
            _speaking.clear()


def speak(text):
    if text:
        print("说：", text, flush=True)
        _say_q.put(text)


def arm(path, body):
    req = urllib.request.Request(ARM_URL + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=3) as r:
            return json.loads(r.read())
    except Exception as e:  # 臂服务没起、或它拒绝了
        print("臂服务：", e, flush=True)
        return {"ok": False, "error": str(e)}


def on_keyword(word, dry=False):
    if not dry:
        arm("/heard", {"text": word})       # 小思页面显示「我听到：…」
    cmd = narrator.parse_command(word)
    if cmd is None:
        print("听到了但大脑不认识：", word)
        return
    path, body, line = cmd
    print(time.strftime("%H:%M:%S"), "听到：", word, "→", path, body, flush=True)
    speak(line)
    if dry:
        return
    r = arm(path, body)
    if not r.get("ok"):
        speak(narrator.LINES.get("refused", "现在不行") + "，" + str(r.get("error", "")))


def _wav_bytes(samples):
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())
    return b.getvalue()


def record_utterance(q, max_s=6.0, wait_s=4.0, tail_s=0.8):
    """叫醒后录一句：等小思说完「我在呢」，声音比底噪大 3 倍算开口，静 0.8 秒算说完；4 秒没开口就算了。"""
    for _ in range(40):                                  # 最多等 4 秒让「我在呢」放完
        if _say_q.empty() and not _speaking.is_set():
            break
        time.sleep(0.1)
    while not q.empty():                                 # 丢掉放声音时录进来的
        q.get_nowait()
    pre, buf, noise, t, quiet = collections.deque(maxlen=3), [], None, 0.0, 0.0
    while t < max_s:
        d = q.get()
        t += len(d) / 16000
        rms = float(np.sqrt(np.mean(d * d)))
        noise = rms if noise is None else noise
        loud = rms > max(0.01, noise * 3)
        if not buf:
            if loud:
                buf = list(pre) + [d]
            elif t > wait_s:
                return None
            else:
                pre.append(d)
                noise = 0.9 * noise + 0.1 * rms
            continue
        buf.append(d)
        quiet = 0.0 if loud else quiet + len(d) / 16000
        if quiet >= tail_s:
            break
    return np.concatenate(buf) if buf else None


def feature_on(name):
    """页面上的功能开关（臂服务 /features）；问不到就当开着。"""
    try:
        with urllib.request.urlopen(ARM_URL + "/features", timeout=1) as r:
            return bool(json.loads(r.read()).get("features", {}).get(name, True))
    except Exception:
        return True


def on_utterance(wav, dry=False):
    """一句话（wav 字节）→ 听写 → 命令就走口令，不是命令就聊天。聊天只说话、上字幕，不调任何动臂的接口。"""
    try:
        text = chat.asr(wav)
    except Exception as e:
        print("听写失败（网络？）：", e, flush=True)
        speak("网络不太好，您用口令叫我吧")
        return None
    print(time.strftime("%H:%M:%S"), "听写：", text or "（没听清）", flush=True)
    if not text:
        speak("我没听清，再说一遍好吗")
        return None
    if not dry:
        arm("/heard", {"text": text})
    target = chat.route(text)
    if target:
        path, body = target
        print("  → 口令", path, body, flush=True)
        if path == "/stop":
            speak("好，停下")
        if dry:
            return target
        r = arm(path, body)
        if not r.get("ok"):
            speak(narrator.LINES.get("refused", "现在不行") + "，" + str(r.get("error", "")))
        return target
    try:
        reply = chat.reply(text)
    except Exception as e:
        print("聊天失败：", e, flush=True)
        reply = "我现在有点卡，您用口令叫我吧"
    print("  → 聊天：", reply, flush=True)
    if not dry:
        arm("/say", {"text": reply})
    speak(reply)
    return reply


def watch_phase(dry=False):
    """盯着臂服务的 phase，变一次说一句。"""
    seen = None
    while True:
        try:
            with urllib.request.urlopen(ARM_URL + "/state", timeout=2) as r:
                s = json.loads(r.read())
        except Exception:
            time.sleep(1.0)
            continue
        seen, line = narrator.line_for(s, seen)
        if line and not dry:
            speak(line)
        time.sleep(0.3)


def make_spotter():
    tag = "epoch-12-avg-2-chunk-16-left-64"
    return sherpa_onnx.KeywordSpotter(
        tokens=str(MODEL / "tokens.txt"),
        encoder=str(MODEL / f"encoder-{tag}.onnx"),
        decoder=str(MODEL / f"decoder-{tag}.onnx"),
        joiner=str(MODEL / f"joiner-{tag}.onnx"),
        keywords_file=str(KEYWORDS),
        num_threads=2, max_active_paths=4,
        keywords_score=1.0, keywords_threshold=0.25, num_trailing_blanks=1,
    )


def run_wav(path):
    spotter = make_spotter()
    with wave.open(path) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1, "要 16k 单声道 wav"
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    s = spotter.create_stream()
    s.accept_waveform(16000, samples)
    s.accept_waveform(16000, np.zeros(int(16000 * 0.5), dtype=np.float32))
    s.input_finished()
    hits = []
    while spotter.is_ready(s):
        spotter.decode_stream(s)
        r = spotter.get_result(s)
        if r:
            hits.append(r)
            spotter.reset_stream(s)
    print(path, "→", hits or "没听到")
    return hits


def run_mic(dry=False, use_chat=True):
    global CHAT
    import sounddevice as sd
    CHAT = use_chat and chat.enabled()
    print("自由对话：" + ("开（叫醒后直接说话）" if CHAT else "关（没有 MiMo 密钥或 --no-chat，只认口令）"), flush=True)
    spotter = make_spotter()
    s = spotter.create_stream()
    threading.Thread(target=_say_worker, daemon=True).start()
    threading.Thread(target=watch_phase, args=(dry,), daemon=True).start()
    q = queue.Queue()
    with sd.InputStream(samplerate=16000, channels=1, dtype="float32",
                        blocksize=1600, callback=lambda d, *_: q.put(d[:, 0].copy())):
        print("在听。说「小思小思」试试。Ctrl+C 退出。", flush=True)
        played = _play["n"]
        while True:
            s.accept_waveform(16000, q.get())
            while spotter.is_ready(s):
                spotter.decode_stream(s)
            r = spotter.get_result(s)
            if _play["n"] != played and r != STOP_WORD:   # 刚放完一段：把放声音时半听进去的清掉（「停下」照收）
                played = _play["n"]
                spotter.reset_stream(s)
                continue
            if r and not accept_keyword(r):
                print("（小思自己在说话，不算）", r, flush=True)
                spotter.reset_stream(s)
                continue
            if r:
                spotter.reset_stream(s)
                on_keyword(r, dry)
                if CHAT and r in WAKE_WORDS and feature_on("chat"):   # 叫醒后听一句自由的话（页面能关）
                    audio = record_utterance(q)
                    if audio is not None:
                        threading.Thread(target=on_utterance, args=(_wav_bytes(audio), dry), daemon=True).start()
                    spotter.reset_stream(s)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", nargs="*", help="用录音文件测，不开麦克风")
    ap.add_argument("--dry", action="store_true", help="只听只说，不调臂服务")
    ap.add_argument("--no-chat", action="store_true", help="不开自由对话，只认口令")
    ap.add_argument("--chat-wav", help="拿一句录音测自由对话（16k 单声道 wav）")
    a = ap.parse_args()
    if a.chat_wav:
        if not chat.enabled():
            sys.exit("没有 MiMo 密钥：设 XIAOSI_MIMO_KEY 或写 ~/.xiaosi_mimo_key")
        CHAT = True
        threading.Thread(target=_say_worker, daemon=True).start()
        on_utterance(pathlib.Path(a.chat_wav).read_bytes(), a.dry)
        time.sleep(0.5)
        while not _say_q.empty() or _speaking.is_set():
            time.sleep(0.2)
        sys.exit(0)
    if a.wav:
        bad = [p for p in a.wav if not run_wav(p)]
        sys.exit(1 if bad else 0)
    run_mic(a.dry, not a.no_chat)
