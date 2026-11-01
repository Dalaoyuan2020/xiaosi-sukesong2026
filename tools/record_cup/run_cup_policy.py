"""带训好的模型跑杯子任务：和录数据时同一套三路相机、同一个起始姿态，臂限速，随时停。

在「终端」App 里跑（要键盘：→ 这条结束，esc 停）：
    cd ~/Documents/sukesong-2026/tools/record_cup
    ./run_cup_policy.sh <模型目录> [--seconds 40] [--episodes 1] [--max-step 0.05]
模型目录 = .../pretrained_model（里面有 config.json、model.safetensors）。

安全：
  - 动作先夹在关节限位以内（每边留 0.05 rad），每一帧每个关节最多动 --max-step rad（默认 0.05 = 每秒 0.75 rad）；
  - 读到 999 / NaN 的动作不发，臂停在上一帧；
  - 夹爪限制在 0–1.6；
  - 5050 孪生还开着就不启动（串口只许一个程序占）。
每次跑会存一个 eval_ 开头的数据集（录下模型怎么动的），不碰训练数据。
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import record_cup as RC  # noqa: E402  只用它的 read_env / VLA / RECORD_FPS / say / twin_alive

JOINT_KEYS = [f"joint{i}.pos" for i in range(1, 7)]
GRIP_KEY = "gripper.pos"
LO = np.array([-2.7, -0.1, -0.1, -2.1, -1.7, -2.7]) + 0.05
HI = np.array([2.7, 3.14, 4.2, 1.6, 1.7, 2.7]) - 0.05


def build_cams(env):
    """和 record_cup.py 一样的三路相机配置（wrist=桥彩色、front=海康 OpenCV、wrist_depth=桥深度）。"""
    cams = []
    for i, role in enumerate(("wrist", "front", "wrist_depth")):
        U = role.upper()
        if role == "wrist_depth" and not env.get(f"{U}_TYPE"):
            continue
        typ = (env.get(f"{U}_TYPE", "opencv") or "opencv").strip().lower()
        w, h = env.get(f"{U}_W", "640"), env.get(f"{U}_H", "480")
        if typ == "bridge":
            url = env[f"{U}_URL"].strip()
            extra = ""
            if env.get(f"{U}_GAMMA", "").strip():
                extra += f", gamma: {env[f'{U}_GAMMA'].strip()}"
            if env.get(f"{U}_TARGET_MEAN", "").strip():
                extra += f", target_mean: {env[f'{U}_TARGET_MEAN'].strip()}"
            cams.append(f"{role}: {{type: bridge, url: '{url}', width: {w}, height: {h}, fps: {RC.RECORD_FPS}{extra}}}")
            continue
        os.environ[f"{U}_CAM_MATCH"] = ""
        idx = env.get(f"{U}_INDEX", str(i))
        cam_fps = int(env.get(f"{U}_FPS", str(RC.RECORD_FPS)) or RC.RECORD_FPS)
        fourcc = env.get(f"{U}_FOURCC", "")
        cams.append(f"{role}: {{type: opencv, index_or_path: {idx}, width: {w}, height: {h}, fps: {cam_fps}"
                    + (f", fourcc: {fourcc}" if fourcc else "") + "}")
    return cams


def install_guard(max_step):
    from lerobot_robot_panthera.panthera_follower import PantheraFollower
    orig = PantheraFollower.send_action
    state = {"prev": None, "clipped": 0, "dropped": 0}

    def send_action(self, action):
        if self.teaching_mode:                       # 回位、示教阶段不管
            return orig(self, action)
        try:
            a = np.array([float(action[k]) for k in JOINT_KEYS])
            g = float(action.get(GRIP_KEY, 1.6))
        except Exception:
            return orig(self, action)
        if not np.all(np.isfinite(a)) or np.any(np.abs(a) >= 999) or not np.isfinite(g):
            state["dropped"] += 1
            if state["dropped"] % 15 == 1:
                print(f"[安全] 模型给了坏动作（999/NaN），不发，第 {state['dropped']} 次", flush=True)
            return action
        if state["prev"] is None:
            try:
                state["prev"] = np.array(self.bus.robot.get_current_pos()[:6], dtype=float)
            except Exception:
                state["prev"] = a.copy()
        b = np.clip(a, LO, HI)
        b = state["prev"] + np.clip(b - state["prev"], -max_step, max_step)
        if np.max(np.abs(b - a)) > 1e-4:
            state["clipped"] += 1
            if state["clipped"] % 30 == 1:
                print(f"[安全] 限速/限位生效（第 {state['clipped']} 帧），最大修正 {np.max(np.abs(b - a)):.3f} rad", flush=True)
        state["prev"] = b
        new = dict(action)
        for k, v in zip(JOINT_KEYS, b):
            new[k] = float(v)
        new[GRIP_KEY] = float(np.clip(g, 0.0, 1.6))
        return orig(self, new)

    PantheraFollower.send_action = send_action
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy", help="训好的模型目录（.../pretrained_model）")
    ap.add_argument("--seconds", type=float, default=40)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--max-step", type=float, default=0.05, help="每帧每个关节最多动多少 rad")
    ap.add_argument("--device", default="mps", help="mps（M 芯片显卡）或 cpu")
    ap.add_argument("--check", action="store_true", help="只解析参数，不连臂")
    ap.add_argument("--task", default=None, help="任务文字（SmolVLA 靠它分辨拿什么，比如 \"pick up the chips\"）；不给就用杯子的")
    a = ap.parse_args()

    pol = Path(a.policy).expanduser().resolve()
    if not (pol / "config.json").exists() or not (pol / "model.safetensors").exists():
        sys.exit(f"{pol} 里没有 config.json / model.safetensors")
    if not a.check and RC.twin_alive():
        sys.exit("5050 孪生还在跑，串口被它占着。先停孪生（扶臂）再跑。")
    device = a.device
    if device == "mps":
        import torch
        if not torch.backends.mps.is_available():
            device = "cpu"

    env = RC.read_env(HERE / "cameras_cup.env")
    cams = build_cams(env)
    import cup_patches as P
    P.apply(use_start_pose=True)
    guard = install_guard(a.max_step)

    name = f"eval_cup_{time.strftime('%m%d_%H%M%S')}"
    # SmolVLA 训练时把三路相机改名成 camera1/2/3（training/train.sh 的 RENAME_MAP），跑的时候要同样改名
    import json as _json
    ptype = _json.loads((pol / "config.json").read_text()).get("type", "")
    rename = ""
    if ptype == "smolvla":
        rename = ('--dataset.rename_map={"observation.images.wrist":"observation.images.camera1",'
                  '"observation.images.front":"observation.images.camera2",'
                  '"observation.images.wrist_depth":"observation.images.camera3"}')
    import lerobot.scripts.lerobot_record as R
    sys.argv = ["lerobot_record.py",
                "--robot.type=panthera_follower",
                "--robot.cameras={ " + ", ".join(cams) + " }",
                f"--dataset.repo_id=local/{name}",
                f"--dataset.root={RC.VLA / 'data'}",
                f"--dataset.single_task={a.task or RC.TASK}",
                f"--dataset.fps={RC.RECORD_FPS}",
                f"--dataset.num_episodes={a.episodes}",
                f"--dataset.episode_time_s={a.seconds}",
                "--dataset.reset_time_s=8",
                "--dataset.push_to_hub=false",
                f"--policy.path={pol}",
                f"--policy.device={device}",
                "--display_data=false",
                "--play_sounds=false"] + ([rename] if rename else [])
    print("\n".join(sys.argv[1:]))
    if a.check:
        return
    print(f"模型：{pol}（{ptype or '?'}）\n设备：{device}　限速：每帧 {a.max_step} rad　时长：{a.seconds}s × {a.episodes}")
    print("按键：→ 这条结束　esc 全部停止。人站在臂旁边，手放急停/电源上。")
    RC.say("小思准备好了，正在回到起始位置")
    R.main()
    RC.say("跑完了")
    print(f"[安全] 限速修正 {guard['clipped']} 帧，丢弃坏动作 {guard['dropped']} 次。录像在 data/local/{name}")


if __name__ == "__main__":
    main()
