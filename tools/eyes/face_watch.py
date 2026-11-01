"""小思的眼睛跟人：用 MacBook 自带摄像头找离屏幕最近的人脸，把位置报给臂服务（POST /look），小思页面让眼珠看过去。

    PY=~/Documents/CodeX_agent/Moonstone_Hackathon/vla/.venv/bin/python
    $PY tools/eyes/face_watch.py --cam 2            # 2 = 内置摄像头（录制用的外接相机是 0/1，别抢）
    $PY tools/eyes/face_watch.py --cam 2 --show     # 带预览窗口调方向

两步：①找人：优先 YuNet（OpenCV 自带接口 + 同目录 230 KB 的模型，侧脸、戴眼镜也稳），没有就退回 Haar；
②看向谁：挑最大的脸（离得最近），别的脸比它大 1.4 倍并持续 0.8 秒才换人，眼睛不来回跳。
只看不录：不存任何画面。x、y 在 -1..1：x>0 = 眼睛往屏幕右边看。
摄像头装在屏幕上方、对着人时默认镜像（--no-mirror 关掉）；--gain / --offset-x 微调，--calibrate 打印坐标对方向。
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

import cv2


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=0.5).read()
    except Exception:
        pass                                           # 臂服务没开也不停，下次再报


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=2)
    ap.add_argument("--url", default="http://127.0.0.1:8000/look")
    ap.add_argument("--hz", type=float, default=8.0)
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--gain", type=float, default=1.3, help="眼睛转动幅度：人在画面边上时眼睛看到多偏")
    ap.add_argument("--offset-x", type=float, default=0.0, help="摄像头不在屏幕正中时的左右补偿（-1..1）")
    ap.add_argument("--calibrate", action="store_true", help="每次检测打印 x，用来对方向")
    a = ap.parse_args()

    model = Path(__file__).with_name("face_detection_yunet_2023mar.onnx")
    yunet = None
    if model.exists() and hasattr(cv2, "FaceDetectorYN"):
        yunet = cv2.FaceDetectorYN.create(str(model), "", (320, 240), 0.7, 0.3, 50)
        print("找人：YuNet", flush=True)
    det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") if hasattr(cv2, "CascadeClassifier") else None
    if yunet is None and det is None:
        raise SystemExit("这个 OpenCV 既没有 YuNet 也没有 Haar：pip install 'opencv-python<5'")
    if yunet is None:
        print("找人：Haar（没找到 YuNet 模型或 OpenCV 太旧）", flush=True)
    cap = cv2.VideoCapture(a.cam)
    if not cap.isOpened():
        raise SystemExit(f"打不开摄像头 {a.cam}：换 --cam 试试（0/1 常是录制用的外接相机）")
    print(f"在看：摄像头 {a.cam} → {a.url}（Ctrl+C 停）", flush=True)
    sx = sy = 0.0
    last_found = 0.0
    target = None            # 当前盯着的脸 (cx, cy, area)
    rival_since = None
    while True:
        t0 = time.time()
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.2)
            continue
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (320, int(320 * h / w)))
        sh, sw = small.shape[:2]
        if yunet is not None:
            yunet.setInputSize((sw, sh))
            _, out = yunet.detect(small)
            faces = [] if out is None else [tuple(int(v) for v in f[:4]) for f in out]
        else:
            gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
            faces = [tuple(f) for f in det.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(24, 24))]
        body = {"found": False, "x": sx, "y": sy}
        if faces:
            cands = [((fx + fw / 2) / sw * 2 - 1, (fy + fh / 2) / sh * 2 - 1, fw * fh, (fx, fy, fw, fh)) for fx, fy, fw, fh in faces]
            big = max(cands, key=lambda c: c[2])
            if target is None:
                target = big
            else:
                near = min(cands, key=lambda c: (c[0] - target[0]) ** 2 + (c[1] - target[1]) ** 2)
                if big is not near and big[2] > 1.4 * near[2]:               # 有人明显更近：持续 0.8 秒才换
                    rival_since = rival_since or time.time()
                    target = big if time.time() - rival_since > 0.8 else near
                else:
                    rival_since, target = None, near
            cx, cy = target[0], target[1]
            if not a.no_mirror:
                cx = -cx
            cx = max(-1.0, min(1.0, (cx + a.offset_x) * a.gain))
            sx, sy = 0.6 * sx + 0.4 * cx, 0.6 * sy + 0.4 * cy           # 平滑，眼睛不抖
            body = {"found": True, "x": round(sx, 3), "y": round(sy, 3)}
            last_found = time.time()
            if a.calibrate:
                print(f"x={sx:+.2f} y={sy:+.2f} faces={len(faces)}", flush=True)
            if a.show:
                fx, fy, fw, fh = target[3]
                cv2.rectangle(small, (fx, fy), (fx + fw, fy + fh), (0, 200, 0), 2)
        else:
            target, rival_since = None, None
        post(a.url, body)
        if a.show:
            cv2.putText(small, f"x={sx:+.2f} y={sy:+.2f} {'FACE' if time.time() - last_found < 1 else '-'}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
            cv2.imshow("xiaosi eyes", small)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        dt = time.time() - t0
        if dt < 1.0 / a.hz:
            time.sleep(1.0 / a.hz - dt)


if __name__ == "__main__":
    main()
