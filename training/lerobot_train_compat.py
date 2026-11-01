"""启动 LeRobot 训练，兼容新版 torchvision。

torchvision 0.25 起删掉了 torchvision.io.VideoReader，LeRobot 0.4.3 用 pyav 读视频时要用它。
魔搭 AMD 环境自带 torchvision 0.27（10/29 实测），一读数据就报错。
这里在缺少 VideoReader 时，把读视频换成直接调用 PyAV，逻辑和 LeRobot 原来的一样：
从第一个时间戳前的关键帧开始解码，取离每个时间戳最近的一帧。
"""

import logging

import torch
import torchvision
from lerobot.datasets import video_utils


def decode_video_frames_pyav(video_path, timestamps, tolerance_s, backend="pyav", log_loaded_timestamps=False):
    import av

    first_ts, last_ts = min(timestamps), max(timestamps)
    loaded_frames, loaded_ts = [], []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        container.seek(int(first_ts / stream.time_base), stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            current_ts = float(frame.pts * stream.time_base)
            if log_loaded_timestamps:
                logging.info(f"frame loaded at timestamp={current_ts:.4f}")
            loaded_frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1))
            loaded_ts.append(current_ts)
            if current_ts >= last_ts:
                break

    query_ts = torch.tensor(timestamps)
    loaded_ts = torch.tensor(loaded_ts)
    min_, argmin_ = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1).min(1)
    is_within_tol = min_ < tolerance_s
    assert is_within_tol.all(), (
        f"查询的时间戳离最近一帧太远（{min_[~is_within_tol]} > {tolerance_s=}）"
        f"\nqueried: {query_ts}\nloaded: {loaded_ts}\nvideo: {video_path}"
    )
    closest_frames = torch.stack([loaded_frames[i] for i in argmin_]).type(torch.float32) / 255
    assert len(timestamps) == len(closest_frames)
    return closest_frames


if not hasattr(torchvision.io, "VideoReader"):
    video_utils.decode_video_frames_torchvision = decode_video_frames_pyav

from lerobot.scripts.lerobot_train import main  # noqa: E402

if __name__ == "__main__":
    main()
