#!/usr/bin/env bash
# 一次性装训练环境。学校超算、4080 的 WSL、魔搭 AMD Notebook 都用这一个：
#   英伟达卡：bash training/setup.sh
#   AMD 卡：  AMD=1 bash training/setup.sh   （不重装 PyTorch，用环境自带的 ROCm 版）
#
# 要能上网（装包、下模型）。超算登录节点上不了网的话：
#   - 模型：从能上网的机器把 $WORK/models/ 整个拷过去
#   - 包：问管理员有没有 pip / conda 镜像
#
# 模型默认从魔搭下载：国内快，π0.5 要的谷歌分词器在魔搭上也不用授权。
# 魔搭不行就改 SOURCE=hf；那时 google/paligemma-3b-pt-224 要先在 Hugging Face 网页上同意许可，
# 再把 token 填进 HF_TOKEN（不要写进仓库）。
set -euo pipefail

# ===== 拿到账号后改这里 =====
# 工作目录：放工作盘或组盘，别放 home（超算 home 一般配额很小）
WORK=${WORK:-/work/$USER/sukesong}
# 录数据用的那份 LeRobot。还是 Panthera-HT 就用默认；换了臂，改成那条臂的 LeRobot 仓库
LEROBOT_REPO=${LEROBOT_REPO:-https://github.com/HighTorque-Robotics/Panthera-HT_lerobot.git}
# π0.5 要的 transformers 改版。GitHub 连不上时换成镜像地址
PI_TRANSFORMERS=${PI_TRANSFORMERS:-git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi}
# ============================

AMD=${AMD:-0}
export HF_HOME=$WORK/cache/hf
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
# pip 镜像。魔搭 / 阿里云机器上清华源会返回 403（10/29 实测），用 PIP_INDEX=https://mirrors.aliyun.com/pypi/simple/
PIP_INDEX=${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}
mkdir -p "$WORK"/cache/hf "$WORK"/models "$WORK"/data "$WORK"/outputs

# 1. Python 环境。AMD Notebook 直接用自带的 Python；其他机器建 conda 环境
#    （有的超算要先 module load anaconda，按学校说明）
if [ "$AMD" != 1 ]; then
  eval "$(conda shell.bash hook)"
  conda env list | grep -q '^lerobot ' || conda create -y -n lerobot python=3.11
  conda activate lerobot
  # Panthera 的 LeRobot fork 锁 torch<2.8；4080 上用的就是 2.6 + cu124
  pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
fi

# 2. LeRobot（Panthera fork 的包在仓库的 lerobot/ 子目录里，原版 LeRobot 在仓库根目录）
SRC=$WORK/$(basename "$LEROBOT_REPO" .git)
[ -d "$SRC" ] || git clone "$LEROBOT_REPO" "$SRC"
if [ -d "$SRC/lerobot/src" ]; then PKG=$SRC/lerobot; else PKG=$SRC; fi

if [ "$AMD" = 1 ]; then
  # LeRobot 要求 torch<2.8，AMD 环境自带的更新。照常装会把 ROCm 版 torch 换成英伟达版，
  # 所以先装除 torch 以外的依赖，再不带依赖地装 LeRobot 本身
  python -c "import tomllib" 2>/dev/null || pip install -i "$PIP_INDEX" tomli   # Python 3.11 起自带
  mapfile -t DEPS < <(python - "$PKG/pyproject.toml" <<'EOF'
import re, sys
try:
    import tomllib
except ImportError:
    import tomli as tomllib
project = tomllib.load(open(sys.argv[1], "rb"))["project"]
extras = project["optional-dependencies"]
deps = project["dependencies"] + [d for d in extras["smolvla"] if not d.startswith("lerobot[")]
for d in deps:
    name = re.split(r"[\[<>=!;~ ]", d, maxsplit=1)[0].lower()
    if name not in ("torch", "torchvision", "torchcodec"):
        print(d)
EOF
)
  pip install -i "$PIP_INDEX" "${DEPS[@]}"
  pip install -i "$PIP_INDEX" --no-deps -e "$PKG"
else
  pip install -i "$PIP_INDEX" -e "$PKG[smolvla]"
fi
pip install -i "$PIP_INDEX" "$PI_TRANSFORMERS"

# 3. 预下载模型：π0.5 底座 + 它的分词器，SmolVLA 底座 + 它的骨干（兜底用）。
#    按 Hugging Face 的名字放进 $WORK/models/，train.sh 在这个目录里运行，代码就直接读本地文件
SOURCE=${SOURCE:-modelscope}
pip install -i "$PIP_INDEX" modelscope
python - "$WORK/models" "$SOURCE" <<'EOF'
import sys
root, source = sys.argv[1], sys.argv[2]
models = {
    "lerobot/pi05_base": None,
    "google/paligemma-3b-pt-224": ["*.json", "*.model"],   # 只要分词器，不下 6GB 的权重
    "lerobot/smolvla_base": None,
    "HuggingFaceTB/SmolVLM2-500M-Video-Instruct": None,
}
if source == "modelscope":
    from modelscope import snapshot_download
else:
    from huggingface_hub import snapshot_download
for repo, patterns in models.items():
    try:
        snapshot_download(repo, local_dir=f"{root}/{repo}", allow_patterns=patterns)
        print(repo, "->", f"{root}/{repo}")
    except Exception as e:
        print(repo, "下载失败：", e)
EOF

# 4. 魔搭镜像的 pi05_base 是新版，处理流程里多了 relative_actions / absolute_actions 两步。
#    LeRobot 0.4.3 不认识它们，训练一启动就报错。这两步在配置里是关闭的，删掉不改变结果；
#    要是遇到开着的未知步骤，就报错停下，不悄悄删
python - "$WORK/models/lerobot/pi05_base" <<'EOF'
import json, pathlib, sys
import lerobot.policies.factory  # noqa: F401  导入后各模型的处理步骤才注册进来
from lerobot.processor import ProcessorStepRegistry
known = set(ProcessorStepRegistry.list())
for f in pathlib.Path(sys.argv[1]).glob("policy_*processor.json"):
    cfg = json.loads(f.read_text())
    keep = []
    for step in cfg["steps"]:
        name = step.get("registry_name")
        if name and name not in known:
            if (step.get("config") or {}).get("enabled", True):
                sys.exit(f"{f.name}: 未知步骤 {name} 是开着的，不能删，先升级 LeRobot")
            print(f"{f.name}: 删掉关闭的未知步骤 {name}")
            continue
        keep.append(step)
    cfg["steps"] = keep
    f.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
EOF

python -c "import torch, lerobot; print('torch', torch.__version__, '| 显卡可用:', torch.cuda.is_available())"
echo "装好了。下一步先冒烟：SMOKE=1 DATA=<数据集名> bash training/train.sh"
