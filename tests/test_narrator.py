"""大脑查表：句子 → 接口，phase → 台词。不起任何服务。"""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from brain import narrator  # noqa: E402


class NarratorTest(unittest.TestCase):
    def test_fixed_sentences(self):
        self.assertEqual(narrator.parse_command("小思小思")[0], "/wake")
        self.assertEqual(narrator.parse_command("小思停下")[0], "/stop")
        self.assertEqual(narrator.parse_command("小思松手")[0], "/release")

    def test_pick_by_alias(self):
        path, body, _ = narrator.parse_command("小思拿药")
        self.assertEqual((path, body["object"]), ("/request", "medicine"))
        self.assertEqual(narrator.parse_command("小思确定")[0], "/confirm")
        self.assertEqual(narrator.parse_command("小思取消")[0], "/cancel")
        self.assertEqual(narrator.parse_command("小思拿杯子")[1]["object"], "cup")
        self.assertEqual(narrator.parse_command("小思拿零食")[1]["object"], "snack")
        self.assertEqual(narrator.parse_command("小思递杯子")[1]["object"], "cup")
        self.assertEqual(narrator.parse_command("小思握手")[:2], ("/request", {"action": "shake"}))
        self.assertEqual(narrator.parse_command("小思同学")[0], "/wake")
        self.assertIsNone(narrator.parse_command("奶龙拿药"))
        self.assertIsNone(narrator.parse_command("小思拿香蕉"))
        self.assertIsNone(narrator.parse_command(""))

    def test_lines_follow_seq(self):
        s = {"seq": 3, "phase": "grasped", "object": "medicine"}
        seq, line = narrator.line_for(s, last_seq=2)
        self.assertEqual(seq, 3)
        self.assertIn("拿到了", line)
        self.assertEqual(narrator.line_for(s, last_seq=3), (3, None))
        seq, line = narrator.line_for({"seq": 4, "phase": "going", "object": "cup"}, 3)
        self.assertIn("水杯", line)
        seq, line = narrator.line_for({"seq": 5, "phase": "failed", "last_result": "超时"}, 4)
        self.assertIn("超时", line)
        seq, line = narrator.line_for({"seq": 6, "phase": "confirming", "object": "shake"}, 5)
        self.assertIn("握握手", line)
        seq, line = narrator.line_for({"seq": 7, "phase": "going", "object": "shake"}, 6)
        self.assertNotIn("去拿", line)


if __name__ == "__main__":
    unittest.main()
