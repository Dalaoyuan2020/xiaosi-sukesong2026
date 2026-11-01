# 小思 · 运行说明（2026 苏客松 · 赛道三）

> 比赛提交版：从团队私有仓库整理出的可公开部分（代码 + 运行说明）。产品页和宣传片：https://hhu.winnielyu.com/sukesong/ 。运行说明 PDF：`RUN.pdf`。

小思是一条会听话的取物机械臂：说「小思小思」叫醒它，说「小思递杯子 / 小思拿零食 / 小思握手」，它先问一遍，你说「小思确定」它才动手；随时说「小思停下」它就停。给行动不便、手够不着的人用。

| 怎么体验 | 地址 / 做法 | 要什么 |
|---|---|---|
| **在线体验**（推荐评委先看） | https://hhu.winnielyu.com/sukesong/demo/ | 一个浏览器 |
| 产品页 + 宣传片 | https://hhu.winnielyu.com/sukesong/ | 一个浏览器 |
| 本地运行（离线备份） | 第 2 节，一条命令 | Python 3.9+ |
| 真机演示 | 第 3 节，现场 | 机械臂 + MacBook + 相机 |

## 1. 在线体验（不用安装）

打开 https://hhu.winnielyu.com/sukesong/demo/ ，这是小思的屏幕页面，后面接的是**假机械臂**（动作用 6 秒的等待代替，不连真臂）。

1. 点〔递杯子〕、〔拿零食〕或〔握手〕：小思先问「要我去拿水杯吗？…然后说：小思确定」，眼睛睁大看着你。
2. 15 秒内点〔确定〕：开始「动手」，屏幕显示进度，做完说「好啦」。不点就作废。
3. 中途点〔停〕：马上停下。
4. 右上角〔模型〕：给每个动作换模型（握手 1 万 / 2 万步，杯子 ACT / SmolVLA，零食合训 / 薯片单训）。

说明：在线版没有麦克风和语音（唤醒词在本地离线识别）；所有访客共用同一个假机械臂，别人点的你也会看到。

## 2. 本地运行（离线备份，不需要机械臂）

```bash
unzip sukesong-2026-xiaosi-src.zip && cd sukesong-2026-xiaosi
python3 -m unittest discover -s tests          # 14 项自测，应全部 OK
ARM_BACKEND=skills XIAOSI_FAKE=1 SKILLS_FILE=config/skills_fake.json python3 arm_service/server.py
# 浏览器打开 http://127.0.0.1:8000/
```
只用 Python 标准库，不用装任何包。和在线体验是同一套代码。

加上语音（macOS，可选）：
```bash
python3 -m venv .venv-voice && .venv-voice/bin/pip install sherpa-onnx==2.0.1 sounddevice numpy click pypinyin sentencepiece
# 唤醒词模型：sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01（sherpa-onnx 官方 releases 下载），解压到 models/
FAKE=1 tools/xiaosi_start.sh                   # 在「终端」App 里跑（麦克风权限跟着终端走）
```
然后对着麦克风说「小思小思」→「小思递杯子」→「小思确定」。自由聊天要小米 MiMo 的 key，写进 `~/.xiaosi_mimo_key`；不写也能用口令。

## 3. 真机运行（现场）

**硬件**：Panthera-HT 六轴机械臂 + 夹爪（USB 串口）；MacBook（M 芯片）；腕上 RealSense D435i（彩色 + 深度）；第三视角 2MP USB 相机；MacBook 自带摄像头（屏幕上的眼睛跟着人看）。

**软件**：Panthera-HT 版 LeRobot 0.4.3（模型推理、控臂）；训练好的模型（ACT / SmolVLA，每个约 0.2–0.7 GB，不放在压缩包里）。

```bash
tools/xiaosi_doctor.sh                          # 一键自检：只看不动臂，逐项 ✅/❌，告诉你缺什么
FAKE=1 tools/xiaosi_start.sh                    # 先用假动作把流程走一遍
tools/skills/shake_model_lab.py --sim           # 握手先仿真：今天的相机 + 3D 孪生回放，过了才许真机
tools/xiaosi_start.sh                           # 正式：人站在臂旁，手放电源键
```
详细的人用说明见 `docs/xiaosi_usage.md`。

**安全线**：动臂时旁边必须有人；只有「小思确定」能让它动手；小思自己说话时只认「小思停下」（不会被自己的台词触发）；一次只许一个程序控臂；读到 999（电机没回话）马上停。

## 4. 代码在哪

| 目录 | 做什么 |
|---|---|
| `arm_service/` | 臂服务：唯一的控制中心。先问再动（/request → /confirm）、停、换模型；每个动作是一个独占串口的技能程序 |
| `config/skills.json` | 动作 → 程序 + 可选模型；`skills_fake.json` 是假动作版 |
| `ui/xiaosi.html` | 小思的屏幕：眼睛、字幕、按钮、模型切换 |
| `voice/` | 离线唤醒词（sherpa-onnx）、台词音频、自由对话入口 |
| `brain/` | 口令解析、台词、自由对话（MiMo 听写 + 对话 + 合成，聊天从不动臂） |
| `tools/skills/` | 握手技能（调测试台跑 ACT 模型，带先仿真闸门） |
| `tools/record_cup/run_cup_policy.*` | 杯子 / 零食技能：LeRobot 带模型跑真臂，限速、坏数据不发 |
| `tools/eyes/` | 找人脸（YuNet）：屏幕眼睛跟人、臂底座跟人 |
| `training/` | 训练脚本 |
| `tests/` | 自测 |

## 5. 训练怎么来的

- 数据：人手带着臂示教，LeRobot 格式，15 fps，三路相机（腕上彩色、腕上深度、第三视角）。递杯子 20 条；零食 80 条（薯片 / 饼干 / 饮料各 20 条，坏的 20 条没用）；握手 47 条（上海场）。
- 模型：ACT（不听指令，一样东西一个模型）和 SmolVLA（听文字指令，比如 "pick up the chips"）。显卡 RTX 4080，batch 8。
- 离线检验（把示教画面喂给模型，比模型给的关节角和人做的差多少）：薯片单训 SmolVLA 4000 步平均差 1.1–1.7°，同条件 ACT 2.4–3.0°；杯子 ACT 1.5–2.2°。都是训练集上的结果，能不能夹起来以真机为准。
