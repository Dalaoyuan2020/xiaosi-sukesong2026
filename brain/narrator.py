"""大脑：把人话变成臂服务的调用，把臂的 phase 变成台词。

现在是纯查表，不联网、不用模型。加分线换大模型时，只换 parse_command() 和 line_for() 的实现，
语音模块（voice/wake.py）调用的方式不变。

    parse_command("小思拿药")  -> ("/request", {"action": "pick", "object": "medicine"}, None)   # 先问，等「小思确定」
    line_for(state, last_seq)   -> (seq, "拿到了，现在递到你手边") 或 (seq, None)

物体和别名在 config/tasks.json，台词在 config/lines.json。
"""

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
LINES = json.loads((ROOT / "config" / "lines.json").read_text())
TASKS = json.loads((ROOT / "config" / "tasks.json").read_text())

NAME = "小思"   # 11/01 改名：思 = speech（会说话）+ 思考。唤醒词「小思小思」

# 固定句子 → (臂服务接口, 请求体, 听到时先说的话)
FIXED = {
    NAME * 2: ("/wake", {}, "我在呢"),             # 叫醒：眼睛看你、说「我在呢」；配置了 wake 技能才动臂
    NAME + "停下": ("/stop", {}, "好，停下"),
    NAME + "松手": ("/release", {}, None),
    NAME + "确定": ("/confirm", {}, None),          # 先问再做：确认后才动臂
    NAME + "取消": ("/cancel", {}, None),
    NAME + "握手": ("/request", {"action": "shake"}, None),   # 先问，等「小思确定」
    NAME + "同学": ("/wake", {}, "我在呢"),       # 队友的人格文件叫「小思同学」，也能叫醒
}


def parse_command(text):
    """一句话 → 调哪个接口。听不懂返回 None。"""
    text = (text or "").strip()
    if text in FIXED:
        return FIXED[text]
    for verb in ("拿", "递"):                    # 「小思拿零食」「小思递杯子」
        if not text.startswith(NAME + verb):
            continue
        want = text[len(NAME) + 1:]
        for key, o in TASKS["objects"].items():
            if want == o["name"] or want in o.get("aliases", []):
                return ("/request", {"action": "pick", "object": key}, None)    # 先问，等「小思确定」
    return None


def line_for(state, last_seq=None):
    """臂服务 /state → 这一步该说的话。seq 没变就不说（返回 None），免得重复。"""
    seq = state.get("seq")
    if seq == last_seq:
        return seq, None
    phase = state.get("phase")
    obj = state.get("object")
    name = LINES.get("objects", {}).get(obj, obj or "东西")
    line = LINES.get("phases", {}).get(phase)
    over = LINES.get("object_phases", {}).get(obj, {})
    if phase in over:                             # 握手这类不是「拿东西」的动作，用自己的台词（null = 不说）
        line = over[phase]
    if phase == "failed" and line is not None:
        line = (line or "") + str(state.get("last_result", ""))
    return seq, (line.replace("{object}", name) if line else None)
