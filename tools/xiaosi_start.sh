#!/bin/bash
# 一键起小思（在 MacBook 的「终端」App 里跑，麦克风和摄像头权限都跟着终端走）：
#   tools/xiaosi_start.sh            # 臂走技能程序（config/skills.json），眼睛用内置摄像头
#   CAM=1 tools/xiaosi_start.sh      # 眼睛用别的摄像头编号（屏幕上方那个）
#   FAKE=1 tools/xiaosi_start.sh     # 假动作演练：说话、页面、按钮、先问再动全走一遍，动作是假的，臂不动（改页面就用它）
#   DRY=1 tools/xiaosi_start.sh      # 只看唤醒词认得对不对：只打印，不调臂服务，页面不会有反应
# 先停掉录制页面和 5050 孪生：串口一次只能一个程序用。Ctrl+C 一起退出。
cd "$(dirname "$0")/.." || exit 1
VLA="${VLA_DIR:-$HOME/Documents/CodeX_agent/Moonstone_Hackathon/vla}"
CAM="${CAM:-2}"
lsof -tiTCP:8768 -sTCP:LISTEN >/dev/null 2>&1 && echo "⚠️  录制页面（8768）还开着，它可能占着串口；先关掉再让小思动臂。"
lsof -tiTCP:5050 -sTCP:LISTEN >/dev/null 2>&1 && echo "⚠️  5050 孪生还开着，它占着串口；skills 后端要先停孪生（扶着臂）。"
trap 'kill 0' INT TERM EXIT
if [ -n "${FAKE:-}" ]; then
  export XIAOSI_FAKE=1 SKILLS_FILE="$PWD/config/skills_fake.json"
  echo "🧪 假动作模式：动作都是假的，臂不会动（config/skills_fake.json）"
fi
ARM_BACKEND=skills python3 arm_service/server.py > logs_xiaosi_arm.log 2>&1 &
sleep 1.5
"$VLA/.venv/bin/python" tools/eyes/face_watch.py --cam "$CAM" > logs_xiaosi_eyes.log 2>&1 &
sleep 1
open -a "Google Chrome" --args --kiosk "http://127.0.0.1:8000/" 2>/dev/null || open "http://127.0.0.1:8000/"
echo "小思起来了：页面 http://127.0.0.1:8000/ （?theme=a 换夜光屏）；臂服务日志 logs_xiaosi_arm.log；眼睛日志 logs_xiaosi_eyes.log"
echo "说「小思小思」叫它，「小思递杯子」→「小思确定」，「小思停下」随时停。"
if [ -n "${DRY:-}" ]; then .venv-voice/bin/python voice/wake.py --dry; else .venv-voice/bin/python voice/wake.py; fi
