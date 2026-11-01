"""小思的自由对话：听写（ASR）→ 判断是不是命令 → 不是命令就用小思的人格聊天 → 说出来（TTS）。

用小米 MiMo（OpenAI 兼容接口）：听写 mimo-v2.5-asr，聊天 mimo-v2.6-flash，说话 mimo-v2.5-tts。
密钥只放在 MacBook 本地：环境变量 XIAOSI_MIMO_KEY，或文件 ~/.xiaosi_mimo_key（chmod 600）。不进仓库、不发微信。
没有密钥时 enabled() 返回 False，语音模块只用离线唤醒词，跟以前一样。

聊天**永远不直接动臂**：要动手的话只能走命令（route() 返回臂服务接口），而且照样先问、等「小思确定」。
"""
import base64
import hashlib
import json
import os
import pathlib
import re
import urllib.request

URL = os.environ.get("XIAOSI_MIMO_URL", "https://token-plan-cn.xiaomimimo.com/v1/chat/completions")
CHAT_MODEL = os.environ.get("XIAOSI_CHAT_MODEL", "mimo-v2.6-flash")
ROOT = pathlib.Path(__file__).resolve().parent.parent
STYLE = "温柔、耐心的年轻女声，像在照顾家里的老人，语速偏慢，咬字清楚，带一点俏皮"


def _persona():
    doc = (ROOT / "docs" / "xiaosi_persona.md").read_text(encoding="utf-8")
    m = re.search(r"## 给大模型的人格提示词.*?```\n(.*?)```", doc, re.S)
    return m.group(1).strip() if m else "你是小思，家里的护理小助手。口语，每次不超过两句。"


def _key():
    k = os.environ.get("XIAOSI_MIMO_KEY", "").strip()
    if not k:
        p = pathlib.Path.home() / ".xiaosi_mimo_key"
        k = p.read_text().strip() if p.exists() else ""
    return k


def enabled():
    return bool(_key())


def _call(body, timeout=30):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + _key(), "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["choices"][0]["message"]


def asr(wav_bytes):
    """16k 单声道 wav → 文字。"""
    msg = _call({"model": "mimo-v2.5-asr", "messages": [{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": base64.b64encode(wav_bytes).decode(), "format": "wav"}}]}]})
    return (msg.get("content") or "").strip()


# 听写出来的一句话 → 臂服务接口（和唤醒词命令同一套：要动臂的一律先 /request，等确认）
# 停 排第一：宁可多停。「确定」只认带「确定/确认」的短句且不带「不」（「我不确定」「可以吗」都不算确认）。
# 「别松手」「不要松开」不算松手。
_ROUTES = [
    (("停下", "停止", "别动", "停"), ("/stop", {})),
    (("取消", "不要了", "算了"), ("/cancel", {})),
    (("松手", "放开", "松开"), ("/release", {})),
    (("握手", "握个手", "握握手"), ("/request", {"action": "shake"})),
    (("杯子", "水杯", "喝水", "拿水", "递水", "杯水", "倒水"), ("/request", {"action": "pick", "object": "cup"})),
    (("零食", "薯片", "饼干", "可乐", "饮料"), ("/request", {"action": "pick", "object": "snack"})),
]
_CONFIRM = ("确定", "确认")


def route(text):
    t = re.sub(r"[\s，。！？、,.!?~～]", "", text or "").removeprefix("小思")
    for words, target in _ROUTES[:3]:
        if any(w in t for w in words):
            if target[0] == "/release" and any(n + w in t for n in ("别", "不要", "不", "先别") for w in words):
                return None
            return target
    if len(t) <= 6 and "不" not in t and any(w in t for w in _CONFIRM):
        return ("/confirm", {})
    for words, target in _ROUTES[3:]:
        if any(w in t for w in words):
            return target
    return None


_history = []


def reply(text):
    """用小思的人格回一句（不超过两句）。记最近 6 轮。"""
    msgs = [{"role": "system", "content": _persona()}] + _history[-12:] + [{"role": "user", "content": text}]
    out = (_call({"model": CHAT_MODEL, "messages": msgs, "max_tokens": 200}).get("content") or "").strip()
    _history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": out}])
    return out or "嗯？我没听清，再说一遍好吗"


def tts_file(text):
    """文字 → 小思的声音（mp3），按文字缓存在 voice/clips/runtime/。失败返回 None（调用方退回系统声音）。"""
    d = ROOT / "voice" / "clips" / "runtime"
    d.mkdir(parents=True, exist_ok=True)
    f = d / (hashlib.md5(text.encode()).hexdigest()[:12] + ".mp3")
    if f.exists():
        return f
    try:
        req = urllib.request.Request(URL, data=json.dumps({
            "model": "mimo-v2.5-tts", "messages": [{"role": "user", "content": STYLE}, {"role": "assistant", "content": text}],
            "audio": {"format": "mp3", "voice": "mimo_default"}}).encode(),
            headers={"Authorization": "Bearer " + _key(), "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.load(r)["choices"][0]["message"]["audio"]["data"]
        f.write_bytes(base64.b64decode(data))
        return f
    except Exception:
        return None
