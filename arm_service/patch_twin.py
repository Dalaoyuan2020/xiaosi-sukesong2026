"""给官方数字孪生后端 app.py 补一个夹爪接口 POST /api/gripper {"position": 1.6}。

    python arm_service/patch_twin.py <app.py 路径>          # 改之前存 app.py.bak；改过的不会重复改
    python arm_service/patch_twin.py <app.py 路径> --check  # 只看能不能改，不写

为什么不直接调 SDK 的 gripper_control()：它会自己发一次串口命令，而后端的控制循环每秒发 200 次，
两个线程同时写串口会串包。所以这里只把夹爪目标存起来，由控制循环在它自己的发送周期里一起发
（SDK 的 motor_send_cmd 会把 7 个电机缓存的命令一起发出去，Panthera.py 的 set_brake 注释写了这点）。

改完要重启后端（twinctl.sh restart）。重启时臂会松力，有人扶着再重启。
"""

import pathlib
import sys

ANCHOR_GLOBAL = "max_torque = [10.0, 10.0, 10.0, 10.0, 10.0, 2.0]\n"
ADD_GLOBAL = ANCHOR_GLOBAL + "gripper_target = None  # 夹爪目标弧度，None = 不碰夹爪（patch_twin.py 加）\n"

ANCHOR_LOOP = "                    robot.pos_vel_MAXtqe(targets, vel, max_torque, iswait=False)\n"
ADD_LOOP = ANCHOR_LOOP + """                    if gripper_target is not None and getattr(robot, 'has_gripper', False):
                        # 只写缓存，下一周期随 6 个关节一起发（patch_twin.py 加）
                        robot.Motors[robot.gripper_id - 1].pos_vel_MAXtqe(gripper_target, 0.5, 0.5)
"""

ANCHOR_ROUTE = "@app.route('/api/set_zero', methods=['POST'])\n"
ADD_ROUTE = '''@app.route('/api/gripper', methods=['POST'])
def set_gripper():
    """夹爪目标（patch_twin.py 加）。1.6 = 张开，0.0 = 合上。只在 position 模式下生效。"""
    global gripper_target
    if robot is None or demo_mode:
        return jsonify({"success": False, "error": "Robot not connected"}), 400
    if not getattr(robot, 'has_gripper', False):
        return jsonify({"success": False, "error": "No gripper"}), 400
    try:
        pos = float(request.json.get('position'))
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "position must be a number"}), 400
    limits = getattr(robot, 'gripper_limits', None) or {"lower": 0.0, "upper": 1.6}
    if not (limits["lower"] <= pos <= limits["upper"]):
        return jsonify({"success": False, "error": f"position out of range {limits}"}), 400
    with target_lock:
        gripper_target = pos
    return jsonify({"success": True, "position": pos})


''' + ANCHOR_ROUTE


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    p = pathlib.Path(sys.argv[1])
    check = "--check" in sys.argv
    raw = p.read_bytes().decode()
    crlf = "\r\n" in raw  # 官方文件可能是 CRLF 换行，照原样写回，免得整个文件都算改过
    s = raw.replace("\r\n", "\n")
    if "/api/gripper" in s:
        print("已经改过了，不动")
        return
    for a in (ANCHOR_GLOBAL, ANCHOR_LOOP, ANCHOR_ROUTE):
        if s.count(a) != 1:
            sys.exit(f"找不到或不止一处，不敢改：{a.strip()[:60]}")
    s = s.replace(ANCHOR_GLOBAL, ADD_GLOBAL).replace(ANCHOR_LOOP, ADD_LOOP).replace(ANCHOR_ROUTE, ADD_ROUTE)
    s = s.replace("    global control_mode, impedance_target\n\n    dt = 1.0 / CONTROL_FREQ",
                  "    global control_mode, impedance_target, gripper_target\n\n    dt = 1.0 / CONTROL_FREQ")
    compile(s, str(p), "exec")
    if check:
        print("能改。")
        return
    bak = p.with_suffix(".py.bak")
    if not bak.exists():
        bak.write_bytes(p.read_bytes())
    p.write_bytes((s.replace("\n", "\r\n") if crlf else s).encode())
    print(f"改好了，备份在 {bak}。重启后端才生效。")


if __name__ == "__main__":
    main()
