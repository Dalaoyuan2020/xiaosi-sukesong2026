# 语音模块

听：sherpa-onnx 关键词检测，离线，不要账号。说：macOS 自带 `say`。臂：只调臂服务 8000 端口。
思必驰授权来不及（10/31 定），先用这套；换 SDK 时只改 `wake.py` 里的 `speak()` 和 `run_mic()`。

## 装（MacBook 上已装好，10/31）

```bash
cd ~/Documents/sukesong-2026
/opt/homebrew/bin/python3.11 -m venv .venv-voice
.venv-voice/bin/pip install sherpa-onnx==2.0.1 sounddevice numpy click pypinyin sentencepiece
mkdir -p models && cd models
curl -sSL -o kws.tar.bz2 https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2
tar xjf kws.tar.bz2 && rm kws.tar.bz2
```

模型 33 MB，不进仓库（`models/` 在 .gitignore）。

## 跑

1. 先起 5050 数字孪生（Moonstone 仓库 `twinctl.sh start`），再起臂服务：`python3 arm_service/server.py`
2. **在 MacBook 的「终端」App 里**（SSH 起的拿不到麦克风）：`.venv-voice/bin/python voice/wake.py`
3. 说「奶龙你好」。

只想听不想动臂：`voice/wake.py --dry`。拿录音测：`voice/wake.py --wav 某.wav`（16k 单声道）。

## 改唤醒词

1. 改 `voice/keywords_raw.txt`，一行一句。每句 4 个字以上，太短会误触发；要 `奶龙停下` 不要 `停`
2. 重新生成拼音：

```bash
M=models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01
.venv-voice/bin/sherpa-onnx-cli text2token --tokens $M/tokens.txt --tokens-type ppinyin voice/keywords_raw.txt voice/keywords.txt
python3 - <<'EOF'
raw=open("voice/keywords_raw.txt").read().split("\n"); tok=open("voice/keywords.txt").read().split("\n")
out=[f"{t.strip()} @{r.strip()}" for r,t in zip(raw,tok) if r.strip() and t.strip() and "@" not in t]
open("voice/keywords.txt","w").write("\n".join(out)+"\n")
EOF
```

每行末尾的 `@奶龙你好` 是识别结果返回的名字，`wake.py` 的 `ACTIONS` 按这个名字找动作。
3. 在 `wake.py` 的 `ACTIONS` 里给新句子配动作。

## 台词

臂每换一个 `phase`，说一句，在 `config/lines.json` 改。`{object}` 会换成物体中文名。

## 10/31 测过的

Mac 合成的 Eddy / Flo / Reed 三种嗓音说「奶龙你好」都认出；「奶龙停下」认出；「今天天气不错我们去吃饭」没误触发。真人、会场噪音还没测。
误触发多就把 `wake.py` 里 `keywords_threshold` 从 0.25 往上调；听不到就往下调。
