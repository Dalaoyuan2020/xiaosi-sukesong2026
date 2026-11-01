"""手拖示教：把臂现在的关节角存成一个有名字的姿态。只读，不会让臂动。

    python arm_service/teach.py save 起始      # 拖到位后存
    python arm_service/teach.py list
    python arm_service/teach.py now            # 只打印当前关节角

先在 5050 页面切到 Gravity 模式，手拖到位、松手等它停稳，再运行 save。
存进 config/poses.json。同名会覆盖，覆盖前打印旧值。
"""

import json
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import twin  # noqa: E402

POSES = pathlib.Path(os.environ.get("POSES_FILE", pathlib.Path(__file__).resolve().parent.parent / "config" / "poses.json"))


def load():
    return json.loads(POSES.read_text()) if POSES.exists() else {}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("save", "list", "now"):
        print(__doc__)
        sys.exit(2)
    cmd = sys.argv[1]

    if cmd == "list":
        for name, p in load().items():
            print(f"{name:12s} {[round(x, 3) for x in p['positions']]}  {p['saved_at']}")
        return

    s = twin.status()
    pos = [round(x, 4) for x in s["positions"]]
    if cmd == "now":
        print(pos)
        return

    if len(sys.argv) < 3:
        sys.exit("要给姿态起个名字：teach.py save <名字>")
    if max(abs(v) for v in s["velocities"]) > 0.05:
        sys.exit("臂还在动，等它停稳再存")
    name = sys.argv[2]
    poses = load()
    if name in poses:
        print(f"覆盖 {name}，旧值 {poses[name]['positions']}")
    poses[name] = {"positions": pos, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    POSES.write_text(json.dumps(poses, ensure_ascii=False, indent=2) + "\n")
    print(f"已存 {name}：{pos}")


if __name__ == "__main__":
    main()
