"""技能启动器：一个技能 = 一个独占串口的外部程序（ACT/SmolVLA 模型跑杯子、轨迹回放拿零食、挥手…）。

为什么这样：模型推理（lerobot_record --policy）、轨迹回放（pure_replay）、录制，都各自要独占机械臂串口。
臂服务不再自己控臂，只做三件事：一次只放一个技能、记 phase 给语音和页面、/stop 时叫停。

技能表在 config/skills.json，只认表里的命令（HTTP 不能传命令，防误操作）：
    {"cup":   {"cmd": ["tools/record_cup/run_cup_policy.sh", "models/cup/pretrained_model", "--seconds", "40"],
               "cwd": "~/Documents/sukesong-2026", "timeout_s": 120},
     "snack": {"cmd": null},            # null = 还没配置，叫它会回「还没配置」
     "wake":  {"cmd": null}}            # 叫「小思小思」时要不要抬一下臂；null = 只出声和眼睛，不动臂
叫停：先发 SIGINT（等价 Ctrl-C，程序自己收尾），5 秒没退再 SIGTERM。整组进程一起发（start_new_session）。
程序的输出逐行写进 logs/skills/<技能>_<时刻>.log；退出码 0 = 完成，其它 = 失败。
"""
import json
import os
import pathlib
import signal
import socket
import subprocess
import threading
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
SKILLS_FILE = pathlib.Path(os.environ.get("SKILLS_FILE", ROOT / "config" / "skills.json"))


def load():
    if not SKILLS_FILE.exists():
        return {}
    return {k: v for k, v in json.loads(SKILLS_FILE.read_text()).items() if not k.startswith("_")}


# 开着就可能占着串口的程序（端口 → 名字）。技能开跑前查一遍，有就拒绝，**不自动收臂**（收臂会动）。
SERIAL_HOLDERS = {8768: "录制页面", 5050: "数字孪生", 8792: "轨迹回放 pure_replay"}
MODEL_LAB = 8790                                       # 握手测试台：连着臂才算占着
FAKE = os.environ.get("XIAOSI_FAKE") == "1"            # 假动作演练（config/skills_fake.json）：不碰串口，不用查


def _listening(port):
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def serial_busy(name):
    """别的程序占着串口就返回原因，没有返回 None。握手技能自己会接着用测试台，不算冲突。"""
    for port, who in SERIAL_HOLDERS.items():
        if _listening(port):
            return f"{who}（{port}）还开着，它可能占着臂。先关掉它再叫我"
    if name != "shake" and _listening(MODEL_LAB):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{MODEL_LAB}/api/state", timeout=2) as r:
                held = json.loads(r.read()).get("arm_connected")
        except Exception:
            held = True                                   # 查不清就当占着
        if held:
            return "握手测试台还抱着臂。先扶住臂，跑 tools/skills/shake_model_lab.py --release 收臂"
    return None


# 模型切换：技能的 cmd 里写 "{model}"，表里 "models": {名字: [参数…]} 列可选模型，"model" 是默认。
# 页面选的存在 logs/skills_choice.json（运行时状态，不进仓库）。
CHOICE_FILE = pathlib.Path(os.environ.get("SKILLS_CHOICE", ROOT / "logs" / "skills_choice.json"))
MODEL_LAB_DIR = pathlib.Path(os.environ.get("MODEL_LAB", "~/Documents/CodeX_agent/Moonstone_Hackathon/model_lab")).expanduser()


def _choices():
    try:
        return json.loads(CHOICE_FILE.read_text())
    except Exception:
        return {}


def current_model(name, spec=None):
    spec = spec if spec is not None else (load().get(name) or {})
    models = spec.get("models") or {}
    if not models:
        return None
    pick = _choices().get(name)
    return pick if pick in models else (spec.get("model") if spec.get("model") in models else next(iter(models)))


def _ready(args):
    """模型文件在不在：pretrained_model 目录看 model.safetensors；--model X 看 model_lab/models/X。"""
    for i, a in enumerate(args):
        if "pretrained_model" in a:
            return (pathlib.Path(os.path.expanduser(a)) / "model.safetensors").exists()
        if i > 0 and args[i - 1] == "--model":
            return (MODEL_LAB_DIR / "models" / a / "model.safetensors").exists()
    return True


def model_list():
    """每个能换模型的技能：可选模型（带「文件在不在」）和现在用的哪个。给页面的模型切换用。"""
    out = {}
    for name, spec in load().items():
        models = spec.get("models") or {}
        if models and spec.get("cmd"):
            out[name] = {"current": current_model(name, spec),
                         "models": [{"name": m, "ready": _ready(a)} for m, a in models.items()]}
    return out


def choose(name, model):
    spec = load().get(name) or {}
    models = spec.get("models") or {}
    if model not in models:
        return False, f"「{name}」没有「{model}」这个模型"
    if not FAKE and not _ready(models[model]):
        return False, f"「{model}」还没下载到这台电脑"
    ch = _choices()
    ch[name] = model
    CHOICE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHOICE_FILE.write_text(json.dumps(ch, ensure_ascii=False, indent=1))
    return True, model


def _expand(name, spec):
    cmd, models = spec.get("cmd") or [], spec.get("models") or {}
    if "{model}" not in cmd:
        return cmd
    args = models.get(current_model(name, spec)) or []
    return [x for c in cmd for x in (args if c == "{model}" else [c])]


QUIET_SKILLS = {"wake"}      # 锦上添花的动作（叫醒点头）：不播台词，做不成也不报失败


def configured(name):
    s = load().get(name)
    return bool(s and s.get("cmd"))


class Runner:
    """跑一个技能程序；on_phase(phase, **kw) 回调给臂服务改 phase。"""

    def __init__(self, on_phase):
        self.on_phase = on_phase
        self.proc = None
        self.stopping = False
        self.model = None

    def start(self, name, object_name=None):
        spec = load().get(name) or {}
        cmd = spec.get("cmd")
        if not cmd:
            return False, f"「{name}」这个技能还没配置（config/skills.json）"
        busy = None if FAKE else serial_busy(name)
        if busy:
            return False, busy
        cwd = pathlib.Path(os.path.expanduser(spec.get("cwd", str(ROOT))))
        log_dir = pathlib.Path(os.environ.get("SKILLS_LOG_DIR", ROOT / "logs" / "skills"))
        log_dir.mkdir(parents=True, exist_ok=True)
        log = open(log_dir / f"{name}_{time.strftime('%m%d_%H%M%S')}.log", "w", encoding="utf-8")
        argv = [os.path.expanduser(c) for c in _expand(name, spec)]
        try:
            self.proc = subprocess.Popen(argv, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True, env=dict(os.environ, PYTHONUNBUFFERED="1"))
        except Exception as e:
            log.close()
            return False, f"启动失败：{e}"
        self.stopping = False
        self.model = current_model(name, spec)
        threading.Thread(target=self._watch, args=(name, object_name, log, float(spec.get("timeout_s", 180))),
                         daemon=True).start()
        return True, str(self.proc.pid)

    def _watch(self, name, object_name, log, timeout_s):
        quiet = name in QUIET_SKILLS
        self.on_phase("waking" if quiet else "going", object=object_name, skill=name, model=self.model, last_result="")
        time.sleep(2.0)
        if self.proc.poll() is None and not quiet:
            self.on_phase("working")
        try:
            rc = self.proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self.stop()
            rc = self.proc.wait()
            self.on_phase("idle" if quiet else "failed", last_result=f"超过 {timeout_s:.0f} 秒没做完")
            log.close()
            return
        log.close()
        if self.stopping:
            self.on_phase("stopped", last_result="stopped")
        elif rc == 0:
            self.on_phase("idle" if quiet else "done", last_result="ok")
        else:
            said = [l.split("小思说：", 1)[1].strip() for l in open(log.name, encoding="utf-8", errors="replace")
                    if l.startswith("小思说：")]      # 技能自己给的、能念给人听的原因
            why = said[-1] if said else f"程序退出码 {rc}，看 logs/skills"
            self.on_phase("idle" if quiet else "failed", last_result=("叫醒动作没做成：" if quiet else "") + why)

    def busy(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        if not self.busy():
            return False
        self.stopping = True
        pg = os.getpgid(self.proc.pid)
        os.killpg(pg, signal.SIGINT)
        for _ in range(50):
            if self.proc.poll() is not None:
                return True
            time.sleep(0.1)
        os.killpg(pg, signal.SIGTERM)
        return True
