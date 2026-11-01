"""小思「扫一圈找到人，再锁定跟着人」：只动机械臂底座（joint1），不需要相机标定。

思路（图像空间视觉伺服）：不去算人在房间里的坐标，只看「人脸在腕上相机画面里偏左还是偏右」。
  1. 扫（SCAN）：底座在起始角度左右各 0.5 rad 内慢慢来回转，腕上相机一直找人脸；
  2. 锁（LOCK）：连续 2 帧看到人脸就停下，改成小步跟：偏左就往左转一点，偏右就往右，到画面中间附近就不动；
     每步最多 0.02 rad（10 Hz = 每秒 0.2 rad），总范围不超过起始 ±0.6 rad；
  3. 人脸丢了 1.5 秒以上，回去扫；
  4. 安全：腕上深度相机中心距离 < 0.40 m（人凑太近）就停住不动；收到 Ctrl-C / SIGINT 或有 /tmp/xiaosi_estop 文件，
     就在当前位置锁住（急停锁定）并退出。
因为是闭环（看一眼、转一点、再看一眼），相机装歪一点、没标定都没关系；唯一要定的是方向（--sign，干跑时看一眼就知道）。

    PY=~/Documents/CodeX_agent/Moonstone_Hackathon/vla/.venv/bin/python
    $PY tools/eyes/arm_follow.py                          # 干跑：只读腕上相机、打印「会怎么转」，不动臂
    $PY tools/eyes/arm_follow.py --source clip.mp4        # 拿录好的视频测找人和控制律
    $PY tools/eyes/arm_follow.py --real --sign 1          # 真动臂（人在旁边、先停录制和孪生、先干跑确认方向）
"""
import argparse
import json
import os
import signal
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
RGB_URL = "http://127.0.0.1:8767/frame.jpg"
INFO_URL = "http://127.0.0.1:8766/info.json"
ESTOP = Path("/tmp/xiaosi_estop")
SCAN_HALF, LOCK_HALF = 0.5, 0.6        # rad：扫的半幅 / 跟的总半幅（相对起始角）
SCAN_SPEED, STEP_MAX = 0.15, 0.02      # rad/s / 每步最多
GAIN, DEADBAND = 0.25, 0.08            # 每单位画面偏移转多少 rad / 中间多大范围不动
TOO_CLOSE_MM = 400


class Eyes:
    """找人脸：YuNet（同目录模型），没有就 Haar。返回最大脸的中心 x（-1..1）和面积占比。"""

    def __init__(self):
        m = HERE / "face_detection_yunet_2023mar.onnx"
        self.yunet = cv2.FaceDetectorYN.create(str(m), "", (320, 240), 0.7, 0.3, 50) if m.exists() and hasattr(cv2, "FaceDetectorYN") else None
        self.haar = (cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                     if hasattr(cv2, "CascadeClassifier") else None)       # OpenCV 5 去掉了 Haar

    def face(self, frame):
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (320, int(320 * h / w)))
        sh, sw = small.shape[:2]
        if self.yunet is not None:
            self.yunet.setInputSize((sw, sh))
            _, out = self.yunet.detect(small)
            faces = [] if out is None else [tuple(int(v) for v in f[:4]) for f in out]
        else:
            faces = [] if self.haar is None else [tuple(f) for f in self.haar.detectMultiScale(
                cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 1.15, 5, minSize=(24, 24))]
        if not faces:
            return None
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        return (fx + fw / 2) / sw * 2 - 1, fw * fh / (sw * sh)


class Source:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src) if src else None

    def frame(self):
        if self.cap is not None:
            ok, f = self.cap.read()
            return f if ok else None
        try:
            with urllib.request.urlopen(RGB_URL, timeout=1.0) as r:
                return cv2.imdecode(np.frombuffer(r.read(), np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            return None


def center_mm():
    try:
        with urllib.request.urlopen(INFO_URL, timeout=0.5) as r:
            return int(json.loads(r.read()).get("center_mm", -1))
    except Exception:
        return -1


class Arm:
    """真臂：Panthera SDK，只改 joint1，其它关节保持起始位置。干跑时只记目标、不发命令。"""

    def __init__(self, real):
        self.real = real
        self.robot = None
        if real:
            sys.path.insert(0, str(HERE.parents[1] / "arm_service"))
            import skills                             # 和技能启动器同一套「谁占着串口」检查
            why = skills.serial_busy("follow")
            if why:
                raise SystemExit(why)
            try:
                with urllib.request.urlopen("http://127.0.0.1:8000/state", timeout=1) as r:
                    if json.loads(r.read()).get("busy"):
                        raise SystemExit("小思正在跑动作技能（臂服务 8000 忙），先停下")
            except OSError:
                pass                                  # 臂服务没开 = 没人占
            from Panthera_lib import Panthera        # SDK：panthera_python/scripts 要在 PYTHONPATH 里
            self.robot = Panthera()
            q = list(self.robot.get_current_pos())
            if any(abs(v) >= 999 for v in q):
                raise SystemExit(f"读到 999：电机没回话，不动臂 {q}")
            self.hold = q[:6]
        else:
            self.hold = [0.0] * 6
        self.q1_0 = self.hold[0]
        self.q1 = self.q1_0

    def move_q1(self, target):
        target = float(np.clip(target, self.q1_0 - LOCK_HALF, self.q1_0 + LOCK_HALF))
        step = float(np.clip(target - self.q1, -STEP_MAX, STEP_MAX))
        self.q1 += step
        if self.real:
            pos = [self.q1] + self.hold[1:]
            self.robot.Joint_Pos_Vel(pos, [0.3] * 6, None, iswait=False)
        return step

    def lock_here(self):
        """急停锁定：在当前位置保持（不松力），然后退出。"""
        if self.real:
            q = list(self.robot.get_current_pos())[:6]
            self.robot.Joint_Pos_Vel(q, [0.2] * 6, None, iswait=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="真动臂（默认干跑）")
    ap.add_argument("--sign", type=float, default=1.0, help="方向：干跑看到人在画面右边时打印的转向和实际要转的方向相反，就改成 -1")
    ap.add_argument("--source", default="", help="测试用视频文件；不填 = 读腕上相机（深度桥 8767）")
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--seconds", type=float, default=0, help="跑多久就停（0 = 一直跑）")
    a = ap.parse_args()

    eyes, src, arm = Eyes(), Source(a.source), Arm(a.real)
    stop = {"now": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    ESTOP.unlink(missing_ok=True)
    print(f"{'真臂' if a.real else '干跑（不动臂）'}：找人用 {'YuNet' if eyes.yunet is not None else 'Haar'}，"
          f"起始 joint1={arm.q1_0:+.3f}，扫 ±{SCAN_HALF} rad，跟 ±{LOCK_HALF} rad", flush=True)
    mode, seen, lost_t, scan_dir, t_end = "SCAN", 0, 0.0, 1.0, (time.time() + a.seconds if a.seconds else None)
    dt = 1.0 / a.hz
    last_print = 0.0
    while not stop["now"] and not ESTOP.exists() and (t_end is None or time.time() < t_end):
        t0 = time.time()
        frame = src.frame()
        if frame is None:
            if a.source:
                break
            time.sleep(0.2)
            continue
        f = eyes.face(frame)
        dist = center_mm() if not a.source else -1
        if f:
            seen, lost_t = seen + 1, 0.0
        else:
            seen = 0
            lost_t = lost_t or time.time()
        if mode == "SCAN" and seen >= 2:
            mode = "LOCK"
        elif mode == "LOCK" and lost_t and time.time() - lost_t > 1.5:
            mode = "SCAN"
        step = 0.0
        if 0 < dist < TOO_CLOSE_MM:
            note = f"太近 {dist} mm，不动"
        elif mode == "SCAN":
            target = arm.q1 + scan_dir * SCAN_SPEED * dt
            if abs(target - arm.q1_0) > SCAN_HALF:
                scan_dir = -scan_dir
            step = arm.move_q1(target)
            note = "扫"
        else:
            x = f[0] if f else 0.0
            err = 0.0 if abs(x) < DEADBAND else x
            step = arm.move_q1(arm.q1 + a.sign * GAIN * err)
            note = f"锁 人脸 x={x:+.2f}" if f else "锁（短暂丢失）"
        if time.time() - last_print > 0.5:
            print(f"{mode} {note} joint1={arm.q1:+.3f} 步={step:+.3f}" + (f" 深度={dist}mm" if dist > 0 else ""), flush=True)
            last_print = time.time()
        if not a.source:
            time.sleep(max(0.0, dt - (time.time() - t0)))
    arm.lock_here()
    print("停下并锁定在当前位置（急停锁定）" if (stop["now"] or ESTOP.exists()) else "结束", flush=True)


if __name__ == "__main__":
    main()
