#!/bin/bash
# 带模型跑杯子任务。在「终端」App 里跑：./run_cup_policy.sh <模型目录> [--seconds 40]
cd "$(dirname "$0")"
VLA="${VLA_DIR:-$HOME/Documents/CodeX_agent/Moonstone_Hackathon/vla}"
export PYTHONPATH="$VLA:$PWD"
exec "$VLA/.venv/bin/python" run_cup_policy.py "$@"
