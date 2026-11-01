"""臂服务对着假 5050 跑一遍：挥手、取物递交、拒绝没示教的物体、中途停。不碰真臂。"""

import json
import os
import pathlib
import socket
import subprocess
import sys
import time
import unittest
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def call(url, body=None):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        with e:
            return e.code, json.loads(e.read())


class ArmServiceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.twin_port, cls.arm_port = free_port(), free_port()
        cls.twin = subprocess.Popen([sys.executable, "arm_service/mock_twin.py", "--port", str(cls.twin_port)],
                                    cwd=ROOT, stdout=subprocess.DEVNULL)
        env = dict(os.environ, TWIN_URL=f"http://127.0.0.1:{cls.twin_port}",
                   POSES_FILE=str(ROOT / "tests" / "poses_fake.json"))
        cls.arm = subprocess.Popen([sys.executable, "arm_service/server.py", "--port", str(cls.arm_port)],
                                   cwd=ROOT, env=env, stdout=subprocess.DEVNULL)
        cls.url = f"http://127.0.0.1:{cls.arm_port}"
        for _ in range(50):
            try:
                call(cls.url + "/state")
                return
            except Exception:
                time.sleep(0.1)
        raise RuntimeError("臂服务没起来")

    @classmethod
    def tearDownClass(cls):
        for p in (cls.arm, cls.twin):
            p.terminate()
            p.wait(timeout=5)

    def state(self):
        return call(self.url + "/state")[1]

    def wait_phase(self, *phases, timeout=20):
        end = time.time() + timeout
        while time.time() < end:
            s = self.state()
            if s["phase"] in phases:
                return s
            time.sleep(0.1)
        self.fail(f"等不到 {phases}，现在是 {self.state()}")

    def test_1_state_is_fake(self):
        s = self.state()
        self.assertEqual(s["mode"], "fake")
        self.assertTrue(s["connected"])
        self.assertFalse(s["busy"])

    def test_2_wave_then_station(self):
        code, r = call(self.url + "/wave", {})
        self.assertEqual(code, 200, r)
        self.assertTrue(r["ok"])
        self.wait_phase("waving", "to_station", "done", timeout=5)
        s = self.wait_phase("done")
        self.assertEqual(s["last_result"], "ok")
        self.assertFalse(s["busy"])

    def test_3_pick_waits_for_release(self):
        code, r = call(self.url + "/pick", {"object": "medicine", "place": "user"})
        self.assertEqual(code, 200, r)
        s = self.wait_phase("waiting_release")
        self.assertEqual(s["object"], "medicine")
        time.sleep(0.5)
        self.assertEqual(self.state()["phase"], "waiting_release", "没说松手不能自己松")
        call(self.url + "/release", {})
        s = self.wait_phase("done")
        self.assertEqual(s["last_result"], "ok")

    def test_4_refuses_untaught_object(self):
        code, r = call(self.url + "/pick", {"object": "cup", "place": "user"})
        self.assertEqual(code, 409)
        self.assertIn("没示教", r["error"])
        code, r = call(self.url + "/pick", {"object": "banana", "place": "user"})
        self.assertEqual(code, 409)

    def test_5_stop_interrupts(self):
        code, r = call(self.url + "/pick", {"object": "medicine", "place": "user"})
        self.assertEqual(code, 200, r)
        self.wait_phase("going", "grasping")
        call(self.url + "/stop", {})
        s = self.wait_phase("stopped", timeout=5)
        self.assertFalse(s["busy"])
        code, r = call(self.url + "/pick", {"object": "medicine", "place": "user"})
        self.assertEqual(code, 200, "停过之后要能再接活")
        call(self.url + "/stop", {})
        self.wait_phase("stopped", timeout=5)


if __name__ == "__main__":
    unittest.main()
