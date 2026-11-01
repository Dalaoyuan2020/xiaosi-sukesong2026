"""把 LeRobot 训练日志画成曲线：上面是 loss，下面是每步用时。

    python training/plot_loss.py <训练日志> <输出图片.png>

日志就是 train.sh 写在 $WORK/outputs/<运行名>.log 的那份，训练中途也能画。
"""

import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_step(s):
    unit = {"K": 1e3, "M": 1e6, "B": 1e9}.get(s[-1])
    return int(float(s[:-1]) * unit) if unit else int(s)


log, out = sys.argv[1], sys.argv[2]
steps, loss, step_s = [], [], []
for line in open(log, encoding="utf-8", errors="ignore"):
    m = re.search(r"step:(\S+) .*loss:([\d.]+).*updt_s:([\d.]+) data_s:([\d.]+)", line)
    if m:
        steps.append(parse_step(m.group(1)))
        loss.append(float(m.group(2)))
        step_s.append(float(m.group(3)) + float(m.group(4)))
if not steps:
    sys.exit("日志里还没有 loss 记录")

fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 7), dpi=110, sharex=True)
a1.plot(steps, loss, color="#1f6feb", lw=1.4, marker="o", ms=3)
a1.set_ylabel("loss")
a1.grid(alpha=0.3)
a1.set_title(f"{log.rsplit('/', 1)[-1]}  |  step {steps[-1]:,}  |  loss {loss[0]:.3f} -> {loss[-1]:.3f}")
a2.plot(steps, step_s, color="#d1242f", lw=1.2, marker="o", ms=3)
a2.set_ylabel("seconds / step")
a2.set_xlabel("training step")
a2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(out)
print(f"{len(steps)} 个点，第 {steps[-1]} 步，loss {loss[-1]:.4f}，每步 {step_s[-1]:.2f} 秒 -> {out}")
