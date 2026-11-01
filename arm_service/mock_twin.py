"""假的 5050：接口和官方数字孪生后端一样，臂是算出来的。开发用，不占串口。

    python arm_service/mock_twin.py --port 5051
    TWIN_URL=http://127.0.0.1:5051 python arm_service/server.py

照着官方 app.py 的脾气做：切 position 不改目标；stop = 目标设成当前位置；
目标按速度匀速逼近。多一个 /api/gripper，和 twin_gripper.patch 加的一样。
状态里多一个 "fake": true，臂服务靠它报 mode=fake。
"""

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

S = {"positions": [0.0] * 6, "target": [0.0] * 6, "velocity": 0.5,
     "mode": "position", "gripper": 1.6}
LOCK = threading.Lock()


def sim():
    dt = 0.02
    while True:
        with LOCK:
            if S["mode"] == "position":
                step = S["velocity"] * dt
                for i in range(6):
                    d = S["target"][i] - S["positions"][i]
                    S["positions"][i] += max(-step, min(step, d))
        time.sleep(dt)


class H(BaseHTTPRequestHandler):
    def _send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        with LOCK:
            if self.path == "/api/status":
                moving = [abs(t - p) > 1e-6 for t, p in zip(S["target"], S["positions"])]
                vel = [S["velocity"] if m and S["mode"] == "position" else 0.0 for m in moving]
                return self._send({"connected": True, "demo_mode": False, "fake": True,
                                   "positions": list(S["positions"]), "velocities": vel,
                                   "torques": [0.0] * 6, "target_positions": list(S["target"]),
                                   "target_velocity": S["velocity"], "gripper": S["gripper"]})
            if self.path == "/api/get_mode":
                return self._send({"mode": S["mode"]})
        self._send({"error": "no"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        d = json.loads(self.rfile.read(n) or b"{}")
        with LOCK:
            if self.path == "/api/move":
                S["target"] = [float(x) for x in d["positions"]]
                if "velocity" in d:
                    S["velocity"] = float(d["velocity"])
            elif self.path == "/api/stop":
                S["target"] = list(S["positions"])
            elif self.path == "/api/set_velocity":
                S["velocity"] = float(d.get("velocity", 0.5))
            elif self.path == "/api/set_mode":
                S["mode"] = d.get("mode", "position")
            elif self.path == "/api/gripper":
                S["gripper"] = float(d["position"])
            elif self.path == "/api/home":
                S["target"] = [0.0] * 6
            else:
                return self._send({"success": False}, 404)
            print(time.strftime("%H:%M:%S"), self.path, d, flush=True)
        self._send({"success": True})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5051)
    a = ap.parse_args()
    threading.Thread(target=sim, daemon=True).start()
    print(f"假 5050 在 http://127.0.0.1:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
