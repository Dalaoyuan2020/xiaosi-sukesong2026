#!/usr/bin/env bash
# 微调 π0.5（主力）、SmolVLA 或 ACT（兜底）。在仓库根目录运行；学校超算上用 train.slurm 提交。
#
#   1. 先冒烟 200 步：查显卡、估速度和费用
#        SMOKE=1 DATA=pick_v1 bash training/train.sh
#   2. 正式训
#        DATA=pick_v1 bash training/train.sh                        # π0.5
#        POLICY=smolvla DATA=pick_v1 bash training/train.sh         # SmolVLA
#        POLICY=act DATA=pick_v1 bash training/train.sh             # ACT
#   3. 魔搭 AMD Notebook 上一样跑；训练时间长的话前面加 nohup，防止网页断开
#
# 数据：把录好的 LeRobot 数据集整个文件夹放到 $WORK/data/<DATA>/
# 权重：训完在 $WORK/outputs/<运行名>/checkpoints/last/pretrained_model
#
# 不默认用 --dataset.episodes 挑条目。要挑的话，Panthera 的 LeRobot fork 必须先打补丁，
# 不然训练看着在收敛、其实没学到动作（MoonStone 踩过）。补丁在 MoonStone 仓库 training/ 里。
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

# ===== 拿到账号后改这里 =====
WORK=${WORK:-/work/$USER/sukesong}                       # 和 setup.sh 一致
# SmolVLA 底座的相机叫 camera1/2/3。把我们录数据时的相机名对上去（名字以数据集里的为准）
RENAME_MAP=${RENAME_MAP:-'{"observation.images.wrist":"observation.images.camera1","observation.images.front":"observation.images.camera2","observation.images.wrist_depth":"observation.images.camera3"}'}
# π0.5 底座按 3 个相机训练。我们录 2 个，就补 1 个空相机
EMPTY_CAMERAS=${EMPTY_CAMERAS:-0}   # 杯子数据集有三路（wrist、front、wrist_depth），π0.5 不用补空相机
# 读数据的进程数。魔搭 AMD 只有 8 核，默认 6；超算 16 核可以调到 12
# ============================

POLICY=${POLICY:-pi05}      # pi05 | smolvla | act
case "$POLICY" in pi05|smolvla|act) ;; *) echo "POLICY 只能是 pi05、smolvla 或 act"; exit 1 ;; esac
DATA=${DATA:?要指定数据集名，例如 DATA=pick_v1}
SMOKE=${SMOKE:-0}
# 默认值来源：
#   π0.5：AMD 博客在 MI300X 上训 π0 用 batch 32；1 万步约 32 万个样本，够我们几十条示教过好几遍
#   SmolVLA、ACT：MoonStone 的留出验证，SmolVLA 2 万步最好，ACT 1 万步最好，再多就过拟合
case "$POLICY" in
  pi05)    STEPS=${STEPS:-10000}; BATCH=${BATCH:-32} ;;
  smolvla) STEPS=${STEPS:-20000}; BATCH=${BATCH:-8} ;;
  act)     STEPS=${STEPS:-10000}; BATCH=${BATCH:-8} ;;
esac
SAVE=${SAVE:-2000}
LOG_FREQ=${LOG_FREQ:-50}
SAVE_CKPT=true
# 冒烟只测能不能跑、多快，不存检查点（π0.5 一个检查点十几 GB，写网盘很慢）
if [ "$SMOKE" = 1 ]; then STEPS=${SMOKE_STEPS:-200}; LOG_FREQ=10; SAVE_CKPT=false; fi

export HF_HOME=$WORK/cache/hf
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}   # 只用 setup.sh 预下载的模型；计算节点一般上不了网
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1   # 日志实时写出来，不然输出被缓存，几分钟看不到进度（魔搭实测）

# 有 setup.sh 建的 lerobot 环境就切过去；魔搭 AMD 没建，直接用自带的 Python
# （超算作业里找不到 conda 的话，先 module load anaconda，按学校说明）
if command -v conda >/dev/null && conda env list | grep -q '^lerobot ' && [ "${CONDA_DEFAULT_ENV:-}" != lerobot ]; then
  eval "$(conda shell.bash hook)"
  conda activate lerobot
fi

RUN=${POLICY}_${DATA}_${SLURM_JOB_ID:-$(date +%m%d_%H%M)}
if [ "$SMOKE" = 1 ]; then RUN=smoke_$RUN; fi

echo "== $RUN | 步数 $STEPS | batch $BATCH | $(date)"
python -c "import torch; assert torch.cuda.is_available(), 'torch 看不到显卡：驱动和 CUDA / ROCm 版本不匹配'; print('torch', torch.__version__, '|', torch.cuda.get_device_name(0))"

case "$POLICY" in
  pi05)
    # 归一化用均值方差：π0.5 默认要分位数统计，我们录的数据集里没有
    POLICY_ARGS=(--policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base
      --policy.dtype=bfloat16
      # 梯度检查点省显存但更慢。魔搭 AMD（192GB）上用 GRAD_CKPT=false：每步 9.4 秒 → 4.9 秒，显存 95%（10/29 实测）
      # 显存小的卡（A40 48GB）必须开着。COMPILE=true 打开编译加速，还没测
      --policy.gradient_checkpointing="${GRAD_CKPT:-true}" --policy.compile_model="${COMPILE:-false}"
      --policy.empty_cameras="$EMPTY_CAMERAS"
      '--policy.normalization_mapping={"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}') ;;
  smolvla)
    POLICY_ARGS=(--policy.path=lerobot/smolvla_base "--rename_map=$RENAME_MAP") ;;
  act)
    POLICY_ARGS=(--policy.type=act) ;;
esac

# setup.sh 把模型按 Hugging Face 的名字放在 $WORK/models/ 下。在这里运行，
# lerobot/pi05_base、google/paligemma-3b-pt-224 这些名字就会读本地文件夹
cd "$WORK/models"
LOG=$WORK/outputs/$RUN.log
# 用兼容入口启动：新版 torchvision 没有 VideoReader 时，改用 PyAV 读视频（魔搭 AMD 实测）
python "$HERE/lerobot_train_compat.py" \
  "${POLICY_ARGS[@]}" \
  --dataset.repo_id=local/"$DATA" \
  --dataset.root="$WORK/data/$DATA" \
  --dataset.video_backend=pyav \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir="$WORK/outputs/$RUN" \
  --job_name="$RUN" \
  --batch_size="$BATCH" \
  --steps="$STEPS" \
  --save_freq="$SAVE" \
  --save_checkpoint="$SAVE_CKPT" \
  --log_freq="$LOG_FREQ" \
  --num_workers="${NUM_WORKERS:-6}" \
  --wandb.enable=false 2>&1 | tee "$LOG"

echo "== 完成 $(date)，权重在 $WORK/outputs/$RUN/checkpoints/last/pretrained_model"
if [ "$SMOKE" = 1 ]; then
  # LeRobot 每 50 步打一行日志，updt_s = 一步训练用时，data_s = 一步读数据用时
  python - "$LOG" "$POLICY" <<'EOF'
import re, sys
text = open(sys.argv[1], errors="ignore").read()
upd = [float(x) for x in re.findall(r"updt_s:([\d.]+)", text)]
dat = [float(x) for x in re.findall(r"data_s:([\d.]+)", text)]
if not upd:
    sys.exit("日志里没找到 updt_s，看上面的报错")
per_step = upd[-1] + (dat[-1] if dat else 0)
print(f"每步约 {per_step:.2f} 秒")
steps = {"pi05": 10000, "smolvla": 20000, "act": 10000}[sys.argv[2]]
print(f"按默认 {steps} 步粗估：约 {per_step * steps / 3600:.1f} 小时")
EOF
fi
