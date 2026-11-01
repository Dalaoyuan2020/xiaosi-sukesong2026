"""臂服务：接口单里的 /pick、/state、/stop，外加 /release（人说「松手」）。

    python arm_service/server.py                        # 真臂：调 5050
    TWIN_URL=http://127.0.0.1:5051 python arm_service/server.py   # 假臂：先起 mock_twin.py

底下的电机层是 5050 数字孪生（见 twin.py），本程序不碰串口。
取物顺序（姿态名在 config/tasks.json，关节角在 config/poses.json）：

    张开夹爪 → 物体上方 → 物体 → 合夹爪 → 物体上方 → 递交点
    → 等人说「松手」（POST /release）→ 张开 → 回起始

每一步改一次 phase，大脑看 /state 的 phase 播报。出错或 /stop 一律停在原地保持（不松力）。
"""

import argparse
import json
import os
import pathlib
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import twin  # noqa: E402
import skills  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
TASKS = json.loads((ROOT / "config" / "tasks.json").read_text())
# 控臂后端：twin = 走 5050 数字孪生 + 示教姿态（原来的做法）；
#          skills = 每个动作是一个独占串口的外部程序（模型推理 / 轨迹回放），表在 config/skills.json
BACKEND = os.environ.get("ARM_BACKEND", "twin")


def load_poses():
    p = pathlib.Path(os.environ.get("POSES_FILE", ROOT / "config" / "poses.json"))
    return json.loads(p.read_text()) if p.exists() else {}


class Stopped(Exception):
    pass


UI_DIR = pathlib.Path(__file__).resolve().parent.parent / "ui"


class Arm:
    def __init__(self):
        self.lock = threading.Lock()
        self.busy = False
        self.abort = threading.Event()
        self.released = threading.Event()
        self.st = {"phase": "idle", "seq": 0, "object": None, "place": None,
                   "task_id": None, "last_result": "", "gripper": None,
                   "heard": "", "heard_at": 0.0,           # 语音模块刚听到的话，小思页面显示
                   "said": "", "said_at": 0.0,             # 自由对话时小思回的话，页面当字幕
                   "look": {"found": False, "x": 0.0, "y": 0.0, "t": 0.0},   # tools/eyes/face_watch.py 报的人脸位置
                   "backend": BACKEND, "skill": None,
                   "pending": None}       # 等确认的动作 {"action","object","expires"}：先确认再执行（安全）
        self.runner = skills.Runner(self.set_phase)

    def set_look(self, body):
        with self.lock:
            self.st["look"] = {"found": bool(body.get("found")), "x": float(body.get("x", 0.0)),
                               "y": float(body.get("y", 0.0)), "t": time.time()}

    def set_heard(self, text):
        with self.lock:
            self.st.update(heard=str(text)[:40], heard_at=time.time())

    def set_said(self, text):
        with self.lock:
            self.st.update(said=str(text)[:80], said_at=time.time())

    def set_phase(self, phase, **kw):
        with self.lock:
            self.st.update(kw, phase=phase, seq=self.st["seq"] + 1)
        print(time.strftime("%H:%M:%S"), phase, kw or "", flush=True)

    def state(self):
        with self.lock:
            pend = self.st.get("pending")
        if pend and time.time() > pend["expires"]:      # 超时没确认：作废
            with self.lock:
                self.st["pending"] = None
            self.set_phase("expired")
        if BACKEND == "skills":
            import glob
            connected, fake = len(glob.glob("/dev/cu.usbmodem2024051701*")) == 7, False   # 臂控制板 7 个串口
            with self.lock:
                return dict(self.st, busy=self.runner.busy(), connected=connected,
                            mode="fake" if skills.FAKE else "real")
        try:
            s = twin.status()
            connected, fake = True, bool(s.get("fake"))
        except Exception:
            connected, fake = False, None
        with self.lock:
            return dict(self.st, busy=self.busy, connected=connected,
                        mode="fake" if fake else "real")

    def run_skill(self, name, object_name=None):
        """skills 后端：一次只跑一个技能程序。"""
        if self.runner.busy():
            return False, "正在干活"
        return self.runner.start(name, object_name)

    CONFIRM_S = 15.0

    def request(self, action, obj=None):
        """先问再做：记下要做的动作，等「小思确定」（或页面「确定」）才执行。"""
        if action not in ("pick", "wave", "shake"):
            return False, f"不认识的动作 {action}"
        if action == "shake":
            obj = "shake"                             # 台词按 object 查：lines.json 的 object_phases.shake
        if action == "pick" and obj not in TASKS["objects"]:
            return False, f"物体清单里没有 {obj}"
        with self.lock:
            self.st["pending"] = {"action": action, "object": obj, "expires": time.time() + self.CONFIRM_S}
        self.set_phase("confirming", object=obj)
        return True, "waiting_confirm"

    def confirm(self):
        with self.lock:
            pend = self.st.get("pending")
            self.st["pending"] = None
        if not pend:
            return False, "没有要确认的事"
        if time.time() > pend["expires"]:
            self.set_phase("expired")
            return False, "确认超时了，要的话再说一遍"
        if pend["action"] == "pick":
            return self.pick(pend["object"], "user")
        if pend["action"] == "shake":
            return self.shake()
        return self.wave()

    def shake(self):
        """握手：只有 skills 后端有（握手模型在 model_lab 测试台里跑，见 tools/skills/shake_model_lab.py）。"""
        if BACKEND != "skills":
            return False, "握手要用 ARM_BACKEND=skills"
        return self.run_skill("shake", "shake")

    def cancel(self):
        with self.lock:
            had = self.st.get("pending") is not None
            self.st["pending"] = None
        if had:
            self.set_phase("cancelled")
        return True, "cancelled"

    def wake(self):
        """「小思小思」：页面和语音负责醒来；配置了 wake 技能（比如抬一下臂）才动臂。"""
        if BACKEND == "skills" and skills.configured("wake"):
            return self.run_skill("wake")
        return True, "awake"

    # 每一步都检查 abort，/stop 能在任何一步打断
    def _check(self):
        if self.abort.is_set():
            raise Stopped()

    def _go(self, pose_name, velocity):
        self._check()
        target = load_poses()[pose_name]["positions"]
        twin.move(target)
        if not twin.wait_arrival(target, velocity, should_abort=self.abort.is_set):
            self._check()
            raise RuntimeError(f"没走到「{pose_name}」（超时）")

    def _gripper(self, pos):
        self._check()
        ok = twin.gripper(pos)
        with self.lock:
            self.st["gripper"] = pos if ok else "无夹爪接口"
        time.sleep(1.0)

    def pick(self, obj, place):
        if BACKEND == "skills":
            return self.run_skill(obj, obj)
        if obj not in TASKS["objects"]:
            return False, f"物体清单里没有 {obj}"
        if place not in TASKS["places"]:
            return False, f"没有这个递交点 {place}"
        o = TASKS["objects"][obj]
        need = [TASKS["start"], o["above"], o["grasp"], TASKS["places"][place]]
        missing = [n for n in need if n not in load_poses()]
        if missing:
            return False, f"这些姿态还没示教：{missing}"
        with self.lock:
            if self.busy:
                return False, "正在干活"
            self.busy = True
        self.abort.clear()
        self.released.clear()
        task_id = uuid.uuid4().hex[:8]
        threading.Thread(target=self._run, args=(task_id, obj, place, o, need), daemon=True).start()
        return True, task_id

    def _run(self, task_id, obj, place, o, need):
        v = TASKS["velocity"]
        start, above, grasp, handover = need
        self.set_phase("going", object=obj, place=place, task_id=task_id, last_result="")
        try:
            twin.hold()
            twin.set_velocity(v)
            self._gripper(1.6)
            self._go(above, v)
            self._go(grasp, v)
            self.set_phase("grasping")
            self._gripper(0.0)
            self.set_phase("grasped")
            self._go(above, v)
            self.set_phase("delivering")
            self._go(handover, v)
            self.set_phase("waiting_release")
            self.released.clear()  # 只认到位之后说的「松手」，路上说的不算
            while not self.released.wait(0.1):
                self._check()
            self.set_phase("releasing")
            self._gripper(1.6)
            self.set_phase("returning")
            self._go(start, v)
            self.set_phase("done", last_result="ok")
        except Stopped:
            self._safe_hold()
            self.set_phase("stopped", last_result="stopped")
        except Exception as e:
            traceback.print_exc()
            self._safe_hold()
            self.set_phase("failed", last_result=str(e))
        finally:
            with self.lock:
                self.busy = False

    def wave(self):
        """唤醒后挥手：挥手1 ↔ 挥手2 来回 3 次，再到「工位」。姿态名在 tasks.json 的 wave 里。"""
        if BACKEND == "skills":
            return self.run_skill("wave")
        w = TASKS["wave"]
        need = [w["a"], w["b"], w["station"]]
        missing = [n for n in need if n not in load_poses()]
        if missing:
            return False, f"这些姿态还没示教：{missing}"
        with self.lock:
            if self.busy:
                return False, "正在干活"
            self.busy = True
        self.abort.clear()
        task_id = uuid.uuid4().hex[:8]
        threading.Thread(target=self._run_wave, args=(task_id, need), daemon=True).start()
        return True, task_id

    def _run_wave(self, task_id, need):
        a, b, station = need
        v = TASKS.get("wave_velocity", TASKS["velocity"])
        self.set_phase("waving", object=None, place=None, task_id=task_id, last_result="")
        try:
            twin.hold()
            twin.set_velocity(v)
            for _ in range(TASKS["wave"].get("times", 3)):
                self._go(a, v)
                self._go(b, v)
            self.set_phase("to_station")
            self._go(station, v)
            self.set_phase("done", last_result="ok")
        except Stopped:
            self._safe_hold()
            self.set_phase("stopped", last_result="stopped")
        except Exception as e:
            traceback.print_exc()
            self._safe_hold()
            self.set_phase("failed", last_result=str(e))
        finally:
            with self.lock:
                self.busy = False

    def _safe_hold(self):
        try:
            twin.hold()
        except Exception:
            traceback.print_exc()

    def stop(self):
        if BACKEND == "skills":
            if not self.runner.stop():
                self.set_phase("stopped", last_result="stopped")
            return
        self.abort.set()
        self._safe_hold()
        if not self.busy:
            self.set_phase("stopped", last_result="stopped")

    def release(self):
        self.released.set()


ARM = Arm()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _file(self, path, ctype):
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/state":
            return self._send(200, ARM.state())
        if self.path.split("?")[0] in ("/", "/xiaosi", "/ui"):   # 小思页面（?theme=a|b|c 选方案）
            return self._file(UI_DIR / "xiaosi.html", "text/html; charset=utf-8")
        if self.path == "/models":                       # 每个动作能选哪些模型、现在用哪个（skills 后端）
            return self._send(200, {"ok": True, "skills": skills.model_list(), "busy": ARM.runner.busy()})
        if self.path == "/lines":
            return self._file(ROOT / "config" / "lines.json", "application/json; charset=utf-8")
        self._send(404, {"ok": False, "error": "没有这个接口"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._send(400, {"ok": False, "error": "不是 JSON"})
        if self.path == "/pick":
            ok, info = ARM.pick(body.get("object"), body.get("place", "user"))
            return self._send(200 if ok else 409, {"ok": ok, "task_id": info} if ok else {"ok": False, "error": info})
        if self.path == "/wave":
            ok, info = ARM.wave()
            return self._send(200 if ok else 409, {"ok": ok, "task_id": info} if ok else {"ok": False, "error": info})
        if self.path == "/stop":
            ARM.stop()
            return self._send(200, {"ok": True})
        if self.path == "/request":                       # 「小思递杯子」：先问，等确认
            ok, info = ARM.request(body.get("action", "pick"), body.get("object"))
            return self._send(200 if ok else 409, {"ok": ok, "info": info} if ok else {"ok": False, "error": info})
        if self.path == "/confirm":                       # 「小思确定」
            ok, info = ARM.confirm()
            return self._send(200 if ok else 409, {"ok": ok, "info": info} if ok else {"ok": False, "error": info})
        if self.path == "/cancel":                        # 「小思取消」
            ok, info = ARM.cancel()
            return self._send(200, {"ok": ok, "info": info})
        if self.path == "/wake":                          # 「小思小思」
            ARM.set_heard(body.get("text", "小思小思"))
            ok, info = ARM.wake()
            return self._send(200 if ok else 409, {"ok": ok, "info": info} if ok else {"ok": False, "error": info})
        if self.path == "/model":                         # 页面换模型：{"skill": "cup", "model": "..."}；干活时不许换
            if ARM.runner.busy():
                return self._send(409, {"ok": False, "error": "正在干活，做完再换"})
            ok, info = skills.choose(body.get("skill", ""), body.get("model", ""))
            return self._send(200 if ok else 409, {"ok": ok, "model": info} if ok else {"ok": False, "error": info})
        if self.path == "/release":
            ARM.release()
            return self._send(200, {"ok": True})
        if self.path == "/look":                         # tools/eyes/face_watch.py：人脸在画面哪儿（-1..1）
            ARM.set_look(body)
            return self._send(200, {"ok": True})
        if self.path == "/heard":                        # voice/wake.py 听到一句就报一下
            ARM.set_heard(body.get("text", ""))
            return self._send(200, {"ok": True})
        if self.path == "/say":                          # voice/wake.py 自由对话：小思回的话，只上字幕，不动臂
            ARM.set_said(body.get("text", ""))
            return self._send(200, {"ok": True})
        self._send(404, {"ok": False, "error": "没有这个接口"})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    print(f"臂服务 http://{a.host}:{a.port}  后端 {BACKEND}"
          + (f"（技能表 {skills.SKILLS_FILE}）" if BACKEND == "skills" else f"（电机层 {twin.TWIN_URL}）"), flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()
