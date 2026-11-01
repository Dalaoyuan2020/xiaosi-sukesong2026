#!/bin/bash
# 小思自检：只看，不动臂、不开串口。逐项打 ✅ / ❌ / ⚠️，❌ 下面一行写怎么补。全部 ✅ 再开 tools/xiaosi_start.sh。
#   tools/xiaosi_doctor.sh
cd "$(dirname "$0")/.." || exit 1
VLA="${VLA_DIR:-$HOME/Documents/CodeX_agent/Moonstone_Hackathon/vla}"
ML="${MODEL_LAB:-$HOME/Documents/CodeX_agent/Moonstone_Hackathon/model_lab}"
RC=tools/record_cup
bad=0
ok() { echo "✅ $1"; }
no() { echo "❌ $1"; echo "     → $2"; bad=$((bad + 1)); }
warn() { echo "⚠️  $1"; echo "     → $2"; }
chk() { if eval "$2" >/dev/null 2>&1; then ok "$1"; else no "$1" "$3"; fi; }
listening() { lsof -tiTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }
GET_CODE='照 tasks/xiaosi_next_local.md 第 0 步，从 LA 拿新代码包解开'
GET_MODEL='模型不在压缩包里：向队伍要，放到 models/<名字>/pretrained_model/'

echo "== 1. 代码"
for f in arm_service/server.py arm_service/skills.py config/skills.json voice/wake.py brain/chat.py \
         tools/skills/shake_model_lab.py ui/xiaosi.html docs/xiaosi_usage.md; do
  chk "$f" "[ -e $f ]" "$GET_CODE"
done
chk "单元测试" "python3 -m unittest discover -s tests" "跑 python3 -m unittest discover -s tests 看哪条错"

echo "== 2. 环境"
chk "模型用的 python（$VLA/.venv）" "[ -x '$VLA/.venv/bin/python' ]" "Moonstone 的 vla 环境不在，VLA_DIR 指到它"
chk "语音 python（.venv-voice，带 sherpa_onnx）" ".venv-voice/bin/python -c 'import sherpa_onnx'" \
    "照 tasks/xiaosi_backbone_local.md 装语音环境"
chk "唤醒词模型" "[ -d models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01 ]" "照 tasks/xiaosi_backbone_local.md 下载"

echo "== 3. 每个动作要的东西（按 config/skills.json）"
python3 - "$ML" <<'EOF'
import json, os, pathlib, sys
ml = pathlib.Path(sys.argv[1])
fix_model = "模型不在压缩包里：向队伍要，放到 models/<名字>/pretrained_model/ ，名字："
cfg = json.load(open("config/skills.json"))
bad = 0
for name, spec in cfg.items():
    if name.startswith("_") or not isinstance(spec, dict):
        continue
    cmd = spec.get("cmd")
    if not cmd:
        print(f"⚠️  {name}：没配（config/skills.json 里 cmd 是 null），这个动作现在用不了")
        continue
    cwd = pathlib.Path(os.path.expanduser(spec.get("cwd", ".")))
    prog = cwd / os.path.expanduser(cmd[0])
    if prog.exists():
        print(f"✅ {name}：程序 {cmd[0]}")
    else:
        bad += 1
        hint = "下载任意一个杯子/零食模型时会顺带装上（cup_down.sh）" if "record_cup" in cmd[0] else "新代码包没解"
        print(f"❌ {name}：程序 {cmd[0]} 不在\n     → {hint}")
    choice = {}
    try:
        choice = json.load(open("logs/skills_choice.json"))
    except Exception:
        pass
    models = spec.get("models") or {}
    cur = choice.get(name) if choice.get(name) in models else spec.get("model")
    runs = [(m, a, m == cur) for m, a in models.items()] if "{model}" in cmd else [(None, cmd, True)]
    for label, args, used in runs:
        for i, arg in enumerate(args):
            p = None
            if "pretrained_model" in arg:
                p = pathlib.Path(os.path.expanduser(arg)) / "model.safetensors"
                dl = pathlib.Path(os.path.expanduser(arg)).parent.name
            elif i > 0 and args[i - 1] == "--model":
                p, dl = ml / "models" / arg / "model.safetensors", None
            if p is None:
                continue
            tag = f"{name}：{label or (p.parent.name if dl is None else dl)}" + ("（现在用的）" if label and used else "")
            if p.exists():
                print(f"✅ {tag}")
            elif used:
                bad += 1
                print(f"❌ {tag} 没下载\n     → " + (fix_model + dl if dl else "model_lab 的握手模型不在，找吕博"))
            else:
                print(f"⚠️  {tag} 没下载（备选，页面上选不了它）\n     → " + (fix_model + dl if dl else "model_lab 里没有"))
sys.exit(bad)
EOF
bad=$((bad + $?))
for f in record_cup.py cameras_cup.env cup_patches.py start_pose.json; do
  chk "杯子/零食共用：$RC/$f" "[ -e $RC/$f ]" "录制程序不全，找 MacBook 上原来的 tools/record_cup"
done
chk "握手测试台 model_lab/server.py" "[ -e '$ML/server.py' ]" "MODEL_LAB 指到 model_lab 目录"
SIM=logs/skills/sim_ok_act_v3_020000.json
if [ -e $SIM ] && python3 -c "import json,time,sys; sys.exit(0 if time.time()-json.load(open('$SIM'))['at'] < 3*3600 else 1)"; then
  ok "握手仿真记录（3 小时内）"
else
  warn "握手还没仿真过（或超过 3 小时），真机会拒绝握手" "浏览器开 http://127.0.0.1:5051，跑 tools/skills/shake_model_lab.py --sim"
fi

echo "== 4. 机械臂和相机（只看在不在）"
chk "机械臂 USB（/dev/cu.usbmodem2024051701*）" "ls /dev/cu.usbmodem2024051701*" "臂没插或没电：查电源和 USB 线"
for p in 8767 8766; do
  chk "腕上深度相机桥 $p" "[ \"\$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:$p/frame.jpg)\" = 200 ]" \
      "深度桥没开：重启 cam-wall-bridge（cam-wall skill 的 depth_bridge.py）"
done
CAMS=$(system_profiler SPCameraDataType 2>/dev/null)
chk "第三视角海康（2MP USB Camera）" "echo \"\$CAMS\" | grep -q '2MP USB Camera'" "海康 USB 线没插好"
chk "FaceTime（眼睛跟人、握手要用）" "echo \"\$CAMS\" | grep -q FaceTime" "内置摄像头被别的程序独占了？"

echo "== 5. 端口（一次只许一个程序占臂）"
if listening 8000; then
  if ps -o command= -p "$(lsof -tiTCP:8000 -sTCP:LISTEN | head -1)" | grep -q mock_arm; then
    no "8000 被假臂页面 ui/dev/mock_arm.py 占着" "关掉它：kill \$(lsof -tiTCP:8000 -sTCP:LISTEN)"
  else
    ok "8000 是小思臂服务（已经在跑）"
  fi
else
  ok "8000 空着（tools/xiaosi_start.sh 会起臂服务）"
fi
for p in 8768:录制页面 5050:数字孪生（连真臂那个） 8792:轨迹回放 8790:握手测试台; do
  listening "${p%%:*}" && warn "${p#*:}（${p%%:*}）开着，会占串口，小思动手会被拒绝" "不用就关掉；8790 是握手停下后没收臂，扶臂跑 tools/skills/shake_model_lab.py --release"
done
listening 5051 && ok "3D 孪生 5051（demo 模式，握手仿真用）" || warn "3D 孪生 5051 没开，握手 --sim 要用" "脚本 --sim 会打印开法"

echo "== 6. 自由对话"
if [ -s ~/.xiaosi_mimo_key ]; then
  [ "$(stat -f %Lp ~/.xiaosi_mimo_key)" = 600 ] && ok "MiMo key 在（不显示内容）" || warn "MiMo key 在，但权限不是 600" "chmod 600 ~/.xiaosi_mimo_key"
else
  warn "没有 MiMo key：自由对话关着，口令照样能用" "吕博自己把 key 写进 ~/.xiaosi_mimo_key，再 chmod 600"
fi

echo
if [ $bad -eq 0 ]; then
  echo "全部必需项 ✅。下一步：DRY=1 tools/xiaosi_start.sh 先练说话（臂不动），见 docs/xiaosi_usage.md"
else
  echo "有 $bad 项 ❌，先按 → 补上再开小思。"
fi
exit $bad
