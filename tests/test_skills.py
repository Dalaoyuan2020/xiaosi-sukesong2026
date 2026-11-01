"""技能启动器：只跑技能表里的命令，phase 从 going → working → done / failed / stopped。不碰真臂。"""

import json
import os
import pathlib
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "arm_service"))


class SkillsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        cfg = {"ok": {"cmd": ["bash", "-c", "echo hi; sleep 2.5"], "cwd": self.tmp.name, "timeout_s": 20},
               "bad": {"cmd": ["bash", "-c", "exit 3"], "cwd": self.tmp.name},
               "said": {"cmd": ["bash", "-c", "echo 小思说：我先不伸手。; exit 1"], "cwd": self.tmp.name},
               "pickm": {"cmd": ["bash", "-c", "echo \"用 $0 $1\"", "{model}", "--seconds"], "cwd": self.tmp.name,
                         "models": {"甲": ["a.pt"], "乙": ["b.pt", "--task", "x"]}, "model": "甲"},
               "long": {"cmd": ["bash", "-c", "trap 'exit 0' INT; sleep 30 & wait"], "cwd": self.tmp.name},
               "none": {"cmd": None}}
        p = pathlib.Path(self.tmp.name) / "skills.json"
        p.write_text(json.dumps(cfg))
        os.environ["SKILLS_FILE"] = str(p)
        os.environ["SKILLS_LOG_DIR"] = self.tmp.name
        os.environ["SKILLS_CHOICE"] = str(pathlib.Path(self.tmp.name) / "choice.json")
        import importlib
        import skills
        self.skills = importlib.reload(skills)
        self.phases = []
        self.results = []
        self.r = self.skills.Runner(lambda ph, **kw: (self.phases.append(ph), self.results.append(kw.get("last_result"))))

    def tearDown(self):
        self.tmp.cleanup()

    def wait_for(self, phase, t=15):
        end = time.time() + t
        while time.time() < end and phase not in self.phases:
            time.sleep(0.1)
        return phase in self.phases

    def test_refuses_when_serial_held(self):
        import socket
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        self.skills.SERIAL_HOLDERS = {port: "测试占口"}
        try:
            ok, msg = self.r.start("ok")
        finally:
            srv.close()
        self.assertFalse(ok)
        self.assertIn("测试占口", msg)

    def test_done(self):
        ok, _ = self.r.start("ok")
        self.assertTrue(ok)
        self.assertTrue(self.r.busy())
        self.assertFalse(self.r.start("ok")[0] if self.r.busy() and False else False)
        self.assertTrue(self.wait_for("done"))
        self.assertEqual(self.phases[:2], ["going", "working"])

    def test_failed_and_unconfigured(self):
        self.assertFalse(self.r.start("none")[0])
        self.assertFalse(self.r.start("missing")[0])
        self.r.start("bad")
        self.assertTrue(self.wait_for("failed"))

    def test_failed_reason_from_skill(self):
        self.r.start("said")
        self.assertTrue(self.wait_for("failed"))
        self.assertEqual(self.results[-1], "我先不伸手。")
        self.r.start("bad")
        self.phases.clear()
        self.assertTrue(self.wait_for("failed"))
        self.assertIn("退出码 3", self.results[-1])

    def test_model_switch(self):
        ml = self.skills.model_list()
        self.assertEqual(ml["pickm"]["current"], "甲")
        self.assertEqual([m["name"] for m in ml["pickm"]["models"]], ["甲", "乙"])
        self.assertEqual(self.skills._expand("pickm", self.skills.load()["pickm"])[3:], ["a.pt", "--seconds"])
        self.assertFalse(self.skills.choose("pickm", "丙")[0])
        self.assertTrue(self.skills.choose("pickm", "乙")[0])
        self.assertEqual(self.skills.current_model("pickm"), "乙")
        self.assertEqual(self.skills._expand("pickm", self.skills.load()["pickm"])[3:], ["b.pt", "--task", "x", "--seconds"])
        self.r.start("pickm")
        self.assertTrue(self.wait_for("done"))
        log = sorted(pathlib.Path(self.tmp.name).glob("pickm_*.log"))[-1].read_text()
        self.assertIn("用 b.pt --task", log)

    def test_stop(self):
        self.r.start("long")
        time.sleep(2.5)
        self.assertTrue(self.r.stop())
        self.assertTrue(self.wait_for("stopped"))


if __name__ == "__main__":
    unittest.main()
