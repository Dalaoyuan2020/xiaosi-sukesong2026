#!/usr/bin/env bash
#SBATCH --job-name=sks-train
#SBATCH --partition=hgpu4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#
# 学校超算上提交训练。参数和 train.sh 一样，在仓库根目录提交：
#   SMOKE=1 DATA=pick_v1 sbatch training/train.slurm
#   DATA=pick_v1 sbatch training/train.slurm
# 看进度：squeue -u $USER；日志在提交目录的 sks-train-<作业号>.out
#
# 队列：π0.5 训练要 48GB 显存，用 normal 档的 A40（hgpu4）；
# 只训 SmolVLA / ACT 的话，可以改成 low 档的 3090（hgpu2）。队列名以学校给的为准。
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
bash training/train.sh
