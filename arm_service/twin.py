"""调 5050 数字孪生后端的小客户端。只用标准库。

5050 是高擎官方的 Panthera_digital_twin（Moonstone 仓库里用 twinctl.sh 起）。
它的后端独占机械臂串口，我们只给它发 HTTP，自己不碰串口（接口单规矩 2）。

用过才知道的三件事（读 app.py 得来）：
- 切到 position 模式时后端不会把目标改成当前位置。目标还是旧值（启动时是全 0），
  臂会自己转过去。所以切模式前一律先 stop（目标 = 当前位置）。见 hold()。
- 后端的路点只存在内存里，最多 6 个，重启就没了。我们的姿态存在 config/poses.json。
- 后端没有夹爪接口，要打补丁（arm_service/twin_gripper.patch）。没打补丁时 gripper() 返回 False。
"""

import json
import math
import os
import time
import urllib.error
import urllib.request

TWIN_URL = os.environ.get("TWIN_URL", "http://127.0.0.1:5050")


class TwinError(RuntimeError):
    pass


def _call(path, body=None, timeout=3.0):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        TWIN_URL + path, data=data, method="GET" if body is None else "POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def status():
    """当前关节角、速度、目标。电机没回话（999）或在 demo 模式就报错。"""
    s = _call("/api/status")
    pos = s.get("positions") or []
    if not s.get("connected") or s.get("demo_mode"):
        raise TwinError("5050 后端没连上臂（demo 模式只有画面，臂不会动）")
    if len(pos) != 6 or any((not math.isfinite(p)) or abs(p) >= 999 for p in pos):
        raise TwinError(f"关节读数不对（999 = 电机没回话）：{pos}")
    return s


def mode():
    return _call("/api/get_mode").get("mode")


def stop():
    """目标 = 当前位置。只在 position 模式下才真的停住；hold() 会顺带切过去。"""
    return _call("/api/stop", {})


def hold():
    """停在原地并保持：先把目标设成当前位置，再切 position。顺序不能反。"""
    stop()
    if mode() != "position":
        _call("/api/set_mode", {"mode": "position"})
        stop()


def set_velocity(v):
    return _call("/api/set_velocity", {"velocity": float(v)})


def move(positions):
    if len(positions) != 6:
        raise TwinError(f"要 6 个关节角，给了 {len(positions)} 个")
    return _call("/api/move", {"positions": [float(p) for p in positions]})


def wait_arrival(target, velocity, tol=0.03, settle_tol=0.10, extra=4.0, should_abort=None):
    """等臂走到 target。

    到位 = 每个关节离目标都在 tol 以内。
    臂带着力矩上限走，重力会让它差一点到不了，所以「停稳了（速度都很小）且误差在 settle_tol 以内」也算到。
    超时 = 最大转角 / 速度 + extra 秒。超时或 should_abort() 为真就返回 False，由调用方去停臂。
    """
    s = status()
    span = max(abs(a - b) for a, b in zip(s["positions"], target))
    deadline = time.time() + span / max(velocity, 0.05) + extra
    still_since = None
    while time.time() < deadline:
        if should_abort and should_abort():
            return False
        s = status()
        err = max(abs(a - b) for a, b in zip(s["positions"], target))
        if err <= tol:
            return True
        if max(abs(v) for v in s["velocities"]) < 0.02 and err <= settle_tol:
            still_since = still_since or time.time()
            if time.time() - still_since > 0.5:
                return True
        else:
            still_since = None
        time.sleep(0.05)
    return False


def gripper(pos):
    """夹爪目标：1.6 = 张开，0.0 = 合上。后端没打补丁时返回 False，不报错。"""
    try:
        return bool(_call("/api/gripper", {"position": float(pos)}).get("success"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise
