#!/usr/bin/env python3
"""小思「握手」技能：调 model_lab 测试台（8790）跑握手 ACT 模型，跑完回零断开、关掉测试台，把串口和相机还回来。

为什么借 model_lab：握手模型（上海场 47 条，ACT v3）就是在它上面上的真机，起始姿态、相机按名字绑定、
限速平滑、「停 = 停在当前位置」都已经在那边测过。这里只当遥控器，不改它的代码。

    tools/skills/shake_model_lab.py                       # 臂服务的 skills 后端调它（config/skills.json 的 shake）
    tools/skills/shake_model_lab.py --check               # 只看测试台起不起得来、模型在不在，不连臂
    tools/skills/shake_model_lab.py --dryrun              # 开真相机跑模型 10 秒，臂不动：看模型给的关节角正不正常
    tools/skills/shake_model_lab.py --release             # 停下之后收臂：回零并断开（会动，先扶住臂）
    tools/skills/shake_model_lab.py --sim                 # 先仿真：真相机试运行 + 在 3D 孪生（5051）里播一遍，过了才许上真机

先仿真后真机：--sim 用今天的相机跑模型（臂不动），自动查关节角没出界、模型有反应、循环够快，再把这段动作
在 5051 的 3D 孪生（demo 模式，不开串口）里播给人看。过了写 logs/skills/sim_ok_<模型>.json。
真机握手只在 3 小时内有同一模型、同一相机分配的仿真记录时才动，否则直接拒绝，说「先跑 --sim」。
注意：试运行时模型看到的关节角一直是起始姿态，也不先转到中间，所以 3D 里看的是「模型对今天的画面反应正不正常」，
不完全等于真机握手的样子。

流程：没开测试台就后台起一个（自己一个进程组，臂服务叫停时不会被连带杀掉）→ 连臂（不动）→
握手（先移到起始姿态约 5 秒，这段停不下来 → 转到中间 2.5 秒 → 模型握 20 秒）→ 回零断开（约 5 秒）→ 关测试台。
叫停（SIGINT，臂服务 /stop 发的）：通知测试台停 → 臂**停在当前位置不松力**，测试台留着抱住臂。
  要收臂：扶住臂，跑 --release（回零并断开），或者在浏览器开 http://127.0.0.1:8790 点「回零并断开」。
退出码：0 握完；1 失败；130 被叫停。
"""
import argparse
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
import urllib.request

ML = pathlib.Path(os.environ.get("MODEL_LAB", "~/Documents/CodeX_agent/Moonstone_Hackathon/model_lab")).expanduser()
VLA = ML.parent / "vla"
STOP = {"now": False}
SIM_TWIN = os.environ.get("SIM_TWIN", "http://127.0.0.1:5051")   # demo 模式的 3D 孪生；绝不用 5050（那个连真臂）
SIM_MAX_AGE_H = 3
NOT_SIMMED = "小思说：握手还没在电脑里先试过，我先不伸手。"     # 臂服务把这句念出来（arm_service/skills.py）
TWIN_START = ("cd ~/Documents/CodeX_agent/Moonstone_Hackathon/vla/Panthera_digital_twin/backend && "
              "python3.11 app.py --demo --config ../../../model_lab/twin_noserial.yaml --port 5051")

# 用测试台的 python（有 numpy）读试运行存下的 runs/<id>.npz：查关节角、模型有没有反应，再在孪生里按 30 Hz 播 cmd。
SIM_PY = r"""
import json, sys, time, urllib.request
import numpy as np
npz, twin, play = sys.argv[1], sys.argv[2].rstrip("/"), sys.argv[3] == "1"
LO = np.array([-2.7, -0.1, -0.1, -2.1, -1.7, -2.7]) - 0.1     # model_lab REAL_LO/HI（Follower.yaml）再放宽 0.1
HI = np.array([2.7, 3.14, 4.2, 1.6, 1.7, 2.7]) + 0.1
d = np.load(npz)
raw, cmd = np.asarray(d["model"])[:, :6], np.asarray(d["cmd"])[:, :6]
out = [f"j{j+1} {raw[:, j].min():+.2f}~{raw[:, j].max():+.2f}" for j in range(6)
       if raw[:, j].min() < LO[j] or raw[:, j].max() > HI[j]]
res = {"frames": len(cmd), "out_of_range": out, "moved_rad": round(float(np.ptp(raw[:, :4], axis=0).max()), 3),
       "cmd_range_deg": [round(float(np.degrees(np.ptp(cmd[:, j]))), 1) for j in range(6)], "played_s": None}
if play:
    def call(path, body=None):
        req = urllib.request.Request(twin + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read() or b"{}")
    if not call("/api/status").get("demo_mode"):
        sys.exit("孪生不是 demo 模式，拒绝回放")
    call("/api/move", {"positions": [float(x) for x in cmd[0]]})
    time.sleep(1.2)
    t0 = time.perf_counter()
    for k, p in enumerate(cmd):
        call("/api/move", {"positions": [float(x) for x in p]})
        time.sleep(max(0.0, t0 + (k + 1) / 30 - time.perf_counter()))
    res["played_s"] = round(time.perf_counter() - t0, 1)
print(json.dumps(res, ensure_ascii=False))
"""


def api(port, path, body=None, timeout=5):
    url = (port.rstrip("/") if isinstance(port, str) else f"http://127.0.0.1:{port}") + path
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                                 method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read())
        except Exception:
            return {"ok": False, "error": f"HTTP {e.code}"}
    except Exception as e:
        return {"ok": False, "error": str(e), "down": True}


def wait(port, ok, seconds, what, stoppable=True):
    """每 0.3 秒看一次 /api/state，ok(state) 为真就返回 state；超时或报错返回 None。"""
    t_end = time.time() + seconds
    last = ""
    while time.time() < t_end:
        if stoppable and STOP["now"]:
            return None
        s = api(port, "/api/state")
        if not s.get("down"):
            if s.get("detail") and s["detail"] != last:
                last = s["detail"]
                print(f"  [{what}] {s.get('state')}：{last}", flush=True)
            if ok(s):
                return s
            if s.get("state") == "error" and not s.get("busy"):
                print(f"测试台报错：{s.get('detail')}", flush=True)
                return None
        time.sleep(0.3)
    print(f"{what}：等了 {seconds:.0f} 秒没好", flush=True)
    return None


def start_lab(port, log_dir):
    """后台起测试台（等于 start.command，但不开浏览器）。自己一个进程组。"""
    py = VLA / ".venv" / "bin" / "python"
    if not (ML / "server.py").exists() or not py.exists():
        sys.exit(f"找不到测试台：{ML}/server.py 或 {py}")
    log_dir.mkdir(parents=True, exist_ok=True)
    log = open(log_dir / f"model_lab_{time.strftime('%m%d_%H%M%S')}.log", "w", encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=str(VLA), WRIST_CAM_MATCH="2MP USB Camera", FRONT_CAM_MATCH="FaceTime",
               PYTORCH_ENABLE_MPS_FALLBACK="1", HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")
    p = subprocess.Popen([str(py), "server.py", "--port", str(port), "--host", "127.0.0.1"], cwd=ML, env=env,
                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(f"起测试台 pid {p.pid}（日志 {log.name}）", flush=True)
    return p


def stop_lab(port, proc):
    """关测试台：只在臂已经断开时关（连着臂时关掉，臂会没力气）。"""
    s = api(port, "/api/state")
    if s.get("arm_connected"):
        print("臂还连着，不关测试台", flush=True)
        return
    pid = proc.pid if proc else None
    if pid is None:
        out = subprocess.run(["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"], capture_output=True, text=True).stdout.split()
        pid = int(out[0]) if out else None
    if pid:
        os.kill(pid, signal.SIGINT)
        for _ in range(30):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.2)
        print("测试台已关，串口和相机已释放", flush=True)


def release(port, proc):
    s = api(port, "/api/state")
    if s.get("down"):
        print("测试台没开，臂没被它占着", flush=True)
        return True
    if s.get("arm_connected"):
        print("回零并断开（约 5 秒，臂会动）…", flush=True)
        r = api(port, "/api/disconnect", {"confirm": True})
        if not r.get("ok"):
            print("回零失败：", r.get("error"), flush=True)
            return False
        if not wait(port, lambda s: not s.get("busy") and not s.get("arm_connected"), 30, "回零", stoppable=False):
            return False
    stop_lab(port, proc)
    return True


def sim_gate(model, cam_map, log_dir):
    """真机前查仿真记录：同一模型、同一相机分配、SIM_MAX_AGE_H 小时内。返回 None = 放行，否则是不放行的原因。"""
    try:
        st = json.loads((log_dir / f"sim_ok_{model}.json").read_text(encoding="utf-8"))
    except Exception:
        return f"没有 {model} 的仿真记录"
    if time.time() - st.get("at", 0) > SIM_MAX_AGE_H * 3600:
        return f"仿真记录是 {st.get('when')} 的，超过 {SIM_MAX_AGE_H} 小时了"
    if cam_map is not None and st.get("cam_map") != cam_map:
        return f"相机分配变了（仿真时 {st.get('cam_map')}，现在 {cam_map}）"
    return None


def sim(a, s, log_dir):
    """先仿真：真相机试运行（臂不动）→ 自动查 → 在 3D 孪生里播。过了写 sim_ok_<模型>.json。"""
    if s.get("arm_connected"):
        print("测试台连着臂，仿真要在没连臂时做（先 --release）", flush=True)
        return False
    r = api(a.port, "/api/dryrun", {"model": a.model, "mode": a.mode, "n_steps": a.n_steps, "duration": a.duration})
    if not r.get("ok"):
        print(f"试运行没启动：{r.get('error')}", flush=True)
        return False
    print(f"仿真 1/2：真相机试运行 {a.duration:.0f} 秒，臂不动", flush=True)
    time.sleep(0.5)
    end = wait(a.port, lambda s: not s.get("busy") and s.get("state") != "running", a.duration + 90, "试运行")
    rows = api(a.port, "/api/results").get("results") or []
    row, m, fails = (rows[-1] if rows else {}), None, []
    if (end or {}).get("state") != "done":
        fails.append(f"试运行没跑完：{(end or {}).get('detail') or '叫停或超时'}")
    elif row.get("model") != a.model or not str(row.get("kind", "")).startswith("试运行"):
        fails.append("对比表最后一行不是这次试运行")
    else:
        if row.get("stopped"):
            fails.append("试运行被叫停了")
        if (row.get("hz") or 0) < 20:          # MacBook 上 async 实测 24.8 Hz（上海场真机也是 24–25）
            fails.append(f"循环只有 {row.get('hz')} Hz（要 ≥ 20）")
        if not a.no_play:
            print(f"仿真 2/2：在 3D 孪生里播这段动作（浏览器开 {SIM_TWIN} 看）", flush=True)
        p = subprocess.run([str(VLA / ".venv" / "bin" / "python"), "-c", SIM_PY, str(ML / "runs" / f"{row['id']}.npz"),
                            SIM_TWIN, "0" if a.no_play else "1"], capture_output=True, text=True, timeout=a.duration + 90)
        try:
            m = json.loads(p.stdout.strip().splitlines()[-1])
        except Exception:
            fails.append("读不出试运行的动作：" + ((p.stderr.strip().splitlines() or ["没有输出"])[-1]))
        if m:
            print(f"  每个关节转了多少度：{m['cmd_range_deg']}；3D 播了 {m['played_s']} 秒", flush=True)
            if m["out_of_range"]:
                fails.append("模型给的关节角出界：" + "；".join(m["out_of_range"]))
            if m["moved_rad"] < 0.05:
                fails.append(f"模型几乎没动（最多 {m['moved_rad']} rad），可能没看懂今天的画面")
    if fails:
        print("仿真没过，不上真机：\n  - " + "\n  - ".join(fails), flush=True)
        return False
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = {"model": a.model, "cam_map": s.get("cam_map"), "at": time.time(), "when": time.strftime("%m-%d %H:%M"),
             "dryrun": row.get("id"), "check": m}
    (log_dir / f"sim_ok_{a.model}.json").write_text(json.dumps(stamp, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"仿真通过（{SIM_MAX_AGE_H} 小时内有效）。3D 里看着像握手，再上真机：扶好臂、旁边有人，跑 tools/skills/shake_model_lab.py",
          flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="act_v3_020000", help="model_lab/models/ 下的名字")
    ap.add_argument("--mode", default="async", choices=["async", "chunk", "te"], help="上海场实机用的 async（分段 + 边执行边算）")
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--duration", type=float, default=20, help="模型控制多少秒")
    ap.add_argument("--start", default="middle", choices=["middle", "side"], help="从起始姿态先转到中间再握")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--keep", action="store_true", help="握完不回零、不关测试台（连着握几次时用）")
    ap.add_argument("--release", action="store_true", help="只做收臂：回零并断开，关测试台")
    ap.add_argument("--check", action="store_true", help="只检查，不连臂")
    ap.add_argument("--dryrun", action="store_true", help="真相机 + 模型跑 --duration 秒，臂不动（不连臂）")
    ap.add_argument("--sim", action="store_true", help="先仿真：试运行 + 自动检查 + 3D 孪生回放，过了才许上真机")
    ap.add_argument("--no-play", action="store_true", help="和 --sim 一起用：只检查，不在孪生里播")
    a = ap.parse_args()
    log_dir = pathlib.Path(os.environ.get("SKILLS_LOG_DIR", pathlib.Path(__file__).resolve().parents[2] / "logs" / "skills"))

    if a.release:
        sys.exit(0 if release(a.port, None) else 1)

    if not (ML / "models" / a.model / "model.safetensors").exists():
        sys.exit(f"模型不在：{ML / 'models' / a.model}")
    real = not (a.check or a.dryrun or a.sim)
    if real and (why := sim_gate(a.model, None, log_dir)):
        print(NOT_SIMMED, flush=True)
        sys.exit(f"不上真机：{why}。先跑仿真：tools/skills/shake_model_lab.py --sim")
    if a.sim and not a.no_play and not api(SIM_TWIN, "/api/status").get("demo_mode"):
        sys.exit(f"3D 孪生 {SIM_TWIN} 没开或不是 demo 模式。开法：{TWIN_START}")

    def on_stop(*_):
        if not STOP["now"]:
            STOP["now"] = True
            print("收到停下：通知测试台停在当前位置", flush=True)
            api(a.port, "/api/stop", {})
    signal.signal(signal.SIGINT, on_stop)
    signal.signal(signal.SIGTERM, on_stop)

    proc = None
    s = api(a.port, "/api/state")
    if s.get("down"):
        proc = start_lab(a.port, log_dir)
        s = wait(a.port, lambda s: True, 90, "等测试台起来")
        if s is None:
            sys.exit(130 if STOP["now"] else 1)
    if s.get("fake"):
        sys.exit("8790 上开的是 --fake 测试台，不是真机，先关掉它")
    print(f"测试台：{s.get('state')}，臂{'已' if s.get('arm_connected') else '未'}连，相机 {s.get('cam_map')}", flush=True)
    if a.check:
        if proc:
            stop_lab(a.port, proc)
        print("检查通过：测试台能起来，模型在", flush=True)
        return
    if a.dryrun:
        if s.get("arm_connected"):
            sys.exit("测试台连着臂，试运行要在没连臂时做（先 --release）")
        r = api(a.port, "/api/dryrun", {"model": a.model, "mode": a.mode, "n_steps": a.n_steps, "duration": a.duration})
        if not r.get("ok"):
            sys.exit(f"试运行没启动：{r.get('error')}")
        end = wait(a.port, lambda s: not s.get("busy") and s.get("state") != "running", a.duration + 90, "试运行")
        rows = api(a.port, "/api/results")
        rows = rows.get("results", rows.get("rows", [])) if isinstance(rows, dict) else rows
        print("试运行结束：", (end or {}).get("state"), (end or {}).get("detail", ""), flush=True)
        if rows:
            print("对比表最后一行：", json.dumps(rows[-1], ensure_ascii=False)[:600], flush=True)
        if proc:
            stop_lab(a.port, proc)
        sys.exit(0 if (end or {}).get("state") == "done" else 1)

    if a.sim:
        ok = sim(a, s, log_dir)
        if proc:
            stop_lab(a.port, proc)
        sys.exit(0 if ok else 1 if not STOP["now"] else 130)
    if why := sim_gate(a.model, s.get("cam_map"), log_dir):
        if proc:
            stop_lab(a.port, proc)
        print(NOT_SIMMED, flush=True)
        sys.exit(f"不上真机：{why}。先跑仿真：tools/skills/shake_model_lab.py --sim")

    if s.get("busy") and not wait(a.port, lambda s: not s.get("busy"), 30, "等测试台空闲"):
        sys.exit(130 if STOP["now"] else 1)
    if not s.get("arm_connected"):
        r = api(a.port, "/api/connect", {})
        if not r.get("ok"):
            print("连臂失败：", r.get("error"), flush=True)
            if proc:
                stop_lab(a.port, proc)
            sys.exit(1)
        if not wait(a.port, lambda s: s.get("arm_connected") and not s.get("busy"), 60, "连臂"):
            if proc and not STOP["now"]:
                stop_lab(a.port, proc)
            sys.exit(130 if STOP["now"] else 1)

    body = {"model": a.model, "mode": a.mode, "n_steps": a.n_steps, "duration": a.duration,
            "start": a.start, "confirm": True}         # confirm = 臂服务那边已经问过人、人说了「小思确定」
    r = api(a.port, "/api/handshake", body)
    if not r.get("ok"):
        print("握手没启动：", r.get("error"), flush=True)
        sys.exit(1)
    print(f"握手：{a.model}（{a.mode}，{a.duration:.0f} 秒）", flush=True)
    time.sleep(0.5)
    end = wait(a.port, lambda s: not s.get("busy") and s.get("state") != "running", a.duration + 40, "握手")
    if STOP["now"]:
        wait(a.port, lambda s: not s.get("busy"), 4, "停下", stoppable=False)
        print("已停：臂停在当前位置，测试台抱着它。收臂：扶住臂，跑 tools/skills/shake_model_lab.py --release", flush=True)
        sys.exit(130)
    state = (end or {}).get("state")
    print(f"握手结束：{state}，{(end or {}).get('detail', '')}", flush=True)
    if state != "done":
        print("没握完，臂停在当前位置。收臂：扶住臂，跑 --release", flush=True)
        sys.exit(1)
    if not a.keep and not release(a.port, proc):
        sys.exit(1)


if __name__ == "__main__":
    main()
