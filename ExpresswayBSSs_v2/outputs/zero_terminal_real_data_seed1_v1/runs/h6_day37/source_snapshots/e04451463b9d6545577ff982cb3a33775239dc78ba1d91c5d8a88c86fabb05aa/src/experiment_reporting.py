"""Export only observed experiment status and completed paired result groups.

This module does not run solvers, infer missing scores, or turn partial days
into complete samples. It may be called repeatedly while a suite is running.
"""
from __future__ import annotations

import csv
from datetime import datetime
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import sys

from .experiment_control import utc_now


def _read(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _atomic_text(path, text, *, csv_file=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig" if csv_file else "utf-8", newline="") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _csv(path, columns, rows):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in columns})
    _atomic_text(path, output.getvalue(), csv_file=True)


def _fmt(value, digits=2):
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}" if math.isfinite(value) else "无效数值"
    return str(value).replace("|", "/").replace("\n", " ")


def _summary(stat, *, scale=1.):
    if not stat or stat.get("mean") is None:
        return "—"
    text = _fmt(stat["mean"] * scale)
    if stat.get("std") is not None:
        text += " ± " + _fmt(stat["std"] * scale)
    return text


def _timestamp(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.timestamp() if parsed.tzinfo is not None else None
    except ValueError:
        return None


def _process_identity_state(pid, created=None, started_at=None):
    """Read-only process evidence; PID reuse means the recorded process ended."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return "unknown"
    try:
        import psutil
    except ImportError:
        return "unknown"
    try:
        process = psutil.Process(pid)
        actual_created = process.create_time()
        if created is not None and abs(actual_created - float(created)) > .01:
            return "pid_reused"
        started = _timestamp(started_at)
        if created is None and started is not None and actual_created > started + .05:
            return "pid_reused"
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            return "exited"
        return "alive"
    except psutil.NoSuchProcess:
        return "exited"
    except (psutil.AccessDenied, OSError, ValueError, TypeError):
        return "unknown"


def _matches_deadline_guard(record, pid, deadline, started_at=None, created=None):
    if (record.get("state") != "paused" or record.get("reason") != "user_wall_clock_deadline"
            or record.get("process_tree_terminated") is not True
            or record.get("worker_pid") != pid):
        return False
    due = _timestamp(deadline)
    guarded_due = _timestamp(record.get("deadline"))
    observed = _timestamp(record.get("observed_at"))
    started = float(created) if created is not None else _timestamp(started_at)
    if (due is None or guarded_due is None or observed is None or abs(due - guarded_due) > .01
            or observed < due or observed > utc_now().timestamp() + 1.
            or due > utc_now().timestamp()):
        return False
    return started is not None and observed >= started


def _controller_context(root):
    status = _read(root / "controller_status.json")
    guard_path = root / "controller_deadline_pause.json"
    guard = _read(guard_path)
    pid = status.get("pid")
    evidence = _process_identity_state(pid, started_at=status.get("started_at"))
    matched = _matches_deadline_guard(guard,pid,status.get("deadline"),status.get("started_at"))
    effective = status.get("state", "not_started")
    source = "controller_status"
    if effective == "running" and evidence in {"exited", "pid_reused"}:
        effective = "paused" if matched else "interrupted_unclassified"
        source = "deadline_guard_and_process_exit" if matched else "process_exit_without_terminal_status"
    return {"status": status, "guard": guard, "guard_path": str(guard_path),
            "guard_matches": matched, "process_state": evidence,
            "state": effective, "state_source": source}


def _effective_worker_state(directory, status, controller):
    process = _read(directory / "process.json")
    raw = status.get("state", "pending")
    pid = process.get("worker_pid", status.get("pid"))
    created = process.get("created")
    started_at = status.get("started_at")
    evidence = _process_identity_state(pid,created,started_at)
    recorded_start = _timestamp(started_at)
    newer_process = bool(process) and (status.get("pid") != pid or
        created is not None and recorded_start is not None and float(created) > recorded_start + .05)
    # A newly resumed live process takes precedence over its predecessor's old
    # paused/complete artifact until the new worker writes its own status.
    state = "running" if newer_process and evidence in {"alive", "unknown"} else raw
    source = ("newer_live_process" if evidence == "alive" else "newer_process_unverified") if state != raw else "worker_status"
    if newer_process and evidence in {"exited", "pid_reused"}:
        state,source = "running","newer_process_without_current_status"
    deadline = process.get("deadline",status.get("deadline"))
    own_path = directory / "deadline_pause.json"
    own_guard = _read(own_path)
    own_match = _matches_deadline_guard(own_guard,pid,deadline,started_at,created)
    parent_match = False
    if controller.get("guard_matches"):
        parent = controller["status"]
        parent_started = _timestamp(parent.get("started_at"))
        worker_started = float(created) if created is not None else recorded_start
        stopped_at = _timestamp(controller["guard"].get("observed_at"))
        parent_match = (parent_started is not None and worker_started is not None
                        and parent_started <= worker_started <= stopped_at
                        and _timestamp(deadline) == _timestamp(parent.get("deadline")))
    guard_path = str(own_path) if own_match else (controller["guard_path"] if parent_match else None)
    if state == "running" and evidence in {"exited", "pid_reused"}:
        state = "paused" if own_match or parent_match else "interrupted_unclassified"
        source = "deadline_guard_and_process_exit" if state == "paused" else "process_exit_without_terminal_status"
    elif state == "running" and evidence == "unknown":
        if source != "newer_process_unverified":
            source = "running_artifact_process_unverified"
    elif state == "running" and evidence == "alive" and guard_path:
        source = "process_still_alive_despite_guard_record"
    return {"state": state,"raw_state": raw,"state_source": source,"process_state": evidence,
            "deadline_evidence_path": guard_path,
            "pause_reason": "user_wall_clock_deadline" if state == "paused" and source == "deadline_guard_and_process_exit"
                            else status.get("reason",status.get("pause_reason"))}


def _pilot_rows(root, plan, comparison, controller):
    known = {str(row["seed"]): row["seed"] for row in comparison.get("pilot_scenarios", [])}
    for seed in plan.get("seeds", {}).get("pilot", []):
        known[str(seed)] = seed
    pilot = root / "pilot"
    if pilot.exists():
        for directory in pilot.glob("seed_*"):
            key = directory.name.removeprefix("seed_")
            known.setdefault(key, int(key) if key.isdigit() else key)
    rows = []
    for key, seed in sorted(known.items()):
        directory = pilot / ("seed_" + key)
        status = _read(directory / "worker_status.json")
        journal = _read(directory / "journal/status.json")
        manifest = _read(directory / "input_manifest.json")
        metrics = status.get("metrics") or _read(directory / "metrics.json")
        target = manifest.get("parameters", {}).get("num_periods")
        effective = _effective_worker_state(directory,status,controller)
        state = effective["state"]
        verified = (state == "complete" and bool(metrics)
                    and isinstance(target, int) and not isinstance(target, bool) and target > 0
                    and metrics.get("completed_periods") == target)
        if state == "complete" and not verified:
            state = "complete_without_verified_metrics"
        periods = metrics.get("completed_periods") if verified else journal.get("completed_periods")
        rows.append({**effective, "seed": seed, "state": state, "completed_periods": periods,
                     "target_periods": target, "verified_full_day": verified,
                     "net_profit_yuan": metrics.get("net_profit_yuan") if verified else None,
                     "worker_wall_seconds": metrics.get("worker_wall_seconds",status.get("wall_seconds")) if verified else None,
                     "error": status.get("error"), "status_path": str(directory / "worker_status.json")})
    return rows


def _environment(root):
    environment = _read(root / "execution_environment.json")
    result = {key: environment.get(key) for key in ("platform", "python", "executable", "cpu_logical", "cpu_physical",
                                                   "memory_bytes", "solver_threads_per_worker", "pilot_process_concurrency")}
    hardware = _read(root / "hardware_environment.json")
    operating_system = hardware.get("operating_system", {})
    result["operating_system"] = " ".join(str(operating_system[key]) for key in ("Caption", "Version")
                                         if operating_system.get(key)) or result.get("platform")
    result["cpu_model"] = "; ".join(row.get("Name", "") for row in hardware.get("cpu", []) if row.get("Name")) or None
    gpu = hardware.get("gpu", {})
    result["gpu_model"] = gpu.get("name")
    result["gpu_memory_mib"] = gpu.get("memory_mib")
    result["training_device_protocol"] = hardware.get("training_device_protocol")
    # Package metadata is inspected only if this exporter uses the same Python
    # executable as the workers. It is not presented as a native solver version.
    same = environment.get("executable") and os.path.normcase(os.path.abspath(environment["executable"])) == os.path.normcase(os.path.abspath(sys.executable))
    result["package_metadata_from_same_worker_python"] = bool(same)
    for package in ("coptpy", "torch"):
        result[package] = None
        if same:
            try:
                result[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
    return result


def export_experiment_report(output_root):
    """Return paths to a Markdown snapshot and four CSV result/status tables."""
    root = Path(output_root).resolve()
    suite_root = root / "formal_suite"
    report = _read(suite_root / "report.json")
    plan = _read(suite_root / "plan.json")
    controller_context = _controller_context(root)
    controller = controller_context["status"]
    comparison = _read(root / "configuration_comparison.json")
    groups = report.get("groups", {})
    completed = {name: group for name, group in groups.items() if group.get("status") == "complete"}
    skipped = [name for name in groups if name not in completed]
    pilot_rows = _pilot_rows(root, plan, comparison, controller_context)
    unfinished_jobs = []
    jobs_root = suite_root / "jobs"
    if jobs_root.exists():
        directories = {path.parent for pattern in ("worker_status.json","process.json") for path in jobs_root.rglob(pattern)}
        for job_directory in sorted(directories):
            status_path = job_directory / "worker_status.json"
            worker_status = _read(status_path)
            effective = _effective_worker_state(job_directory,worker_status,controller_context)
            if effective["state"] == "complete":
                continue
            journal = _read(job_directory / "journal/status.json")
            unfinished_jobs.append({"job_id":str(job_directory.relative_to(jobs_root)).replace(os.sep,"/"),
                **effective,"completed_periods":journal.get("completed_periods"),
                "error":worker_status.get("error"),"status_path":str(status_path)})
    environment = _environment(root)
    rows, paired = [], []
    for name, group in sorted(completed.items()):
        seeds = group.get("scenario_seeds", [])
        if len(seeds) != len(set(seeds)):
            raise ValueError("a completed group has duplicate test scenario seeds: " + name)
        metadata = group.get("metadata", {})
        base = {"group": name, "training_replicate": metadata.get("training_replicate"),
                "kind": metadata.get("kind"), "feature_variant": metadata.get("variant"),
                "scenario_count": len(seeds), "scenario_seeds": ";".join(map(str,seeds))}
        for metric, stats in sorted(group.get("statistics", {}).items()):
            if stats.get("n", 0) > len(seeds):
                raise ValueError("statistical n exceeds the distinct test scenarios: " + name)
            rows.append({**base, "metric": metric, **stats})
        for metric, stats in sorted(group.get("paired_difference_from_zero", {}).items()):
            if stats.get("n", 0) > len(seeds):
                raise ValueError("paired n exceeds the distinct test scenarios: " + name)
            paired.append({**base, "metric": metric, "reference": "paired zero-terminal group", **stats})
    directory = root / "reports"
    paths = {"markdown": directory / "experiment_report.md",
             "completed_groups_csv": directory / "completed_groups.csv",
             "paired_differences_csv": directory / "paired_differences.csv",
             "pilot_status_csv": directory / "pilot_status.csv",
             "unfinished_jobs_csv": directory / "unfinished_jobs.csv"}
    columns = ["group","training_replicate","kind","feature_variant","scenario_count","scenario_seeds",
               "metric","n","mean","std","ci95_low","ci95_high"]
    _csv(paths["completed_groups_csv"],columns,rows)
    _csv(paths["paired_differences_csv"],columns+["reference"],paired)
    evidence_columns = ["raw_state","state_source","process_state","pause_reason","deadline_evidence_path"]
    _csv(paths["pilot_status_csv"],["seed","state","completed_periods","target_periods","verified_full_day",
                                  "net_profit_yuan","worker_wall_seconds","error","status_path"]+evidence_columns,pilot_rows)
    _csv(paths["unfinished_jobs_csv"],["job_id","state","completed_periods","error","status_path"]+evidence_columns,unfinished_jobs)
    status = report.get("status", "not_started")
    active_processes = any(row["state"] == "running" for row in pilot_rows+unfinished_jobs)
    if status == "running" and controller_context["state"] == "paused" and not active_processes:
        status = "paused"
    lines = ["# 实验进度与已完成结果", "", "生成时间（UTC）：" + utc_now().isoformat(), "",
             "控制器状态：" + controller_context["state"] + "；正式套件状态：" + status + "。",
             "本报告仅汇总已完成的整日分组。部分轨迹、暂停和失败记录不计入正式样本数，不用零补齐结果。", ""]
    lines += ["状态口径：running 表示仍运行或运行记录尚未核验；paused 表示主动暂停，或截止 guard 记录与原进程已退出共同证实的暂停；failed 保留明确失败；pending/not_started 表示未开始；complete 仅在整日记录核验后计收益。",
              "interrupted_unclassified 表示原进程已结束但没有足够证据区分暂停与失败；不将其计为完成。PID 被复用时，按原进程身份判断。", ""]
    if controller_context["state_source"] == "deadline_guard_and_process_exit":
        lines += ["控制器暂停由 controller_deadline_pause.json 与原进程已退出的证据核验；状态文件中的旧 running 未作为当前运行事实。", ""]
    if unfinished_jobs:
        lines += ["## 未完成的正式任务", "", "| 任务 | 当前状态 | 已完成周期 | 状态依据 |",
                  "|---|---|---:|---|"]
        for row in unfinished_jobs[:30]:
            lines.append("| " + row["job_id"] + " | " + row["state"] + " | " + _fmt(row["completed_periods"],0) + " | " + row["state_source"] + " |")
        lines += ["", "完整未完成任务及暂停证据见 unfinished_jobs.csv；这些任务不进入正式收益统计。", ""]
    if controller.get("deadline"):
        lines.append("本次运行截止时刻：" + str(controller["deadline"]) + "。截止时保留待续任务及原种子，不缩减冻结预算。")
        lines.append("")
    lines += ["## 预实验", "", "预实验仅用于可行性检查和计时，不作为正式测试样本。", "",
              "| 种子 | 状态 | 完成周期/目标 | 整日净收益/元 | 整日耗时/s |",
              "|---|---|---|---:|---:|"]
    for row in pilot_rows:
        progress = _fmt(row["completed_periods"],0) + "/" + _fmt(row["target_periods"],0)
        lines.append("| " + str(row["seed"]) + " | " + row["state"] + " | " + progress + " | "
                     + _fmt(row["net_profit_yuan"]) + " | " + _fmt(row["worker_wall_seconds"]) + " |")
    for row in pilot_rows:
        if row.get("error"):
            lines.append("")
            lines.append("种子 " + str(row["seed"]) + " 的记录错误：" + _fmt(row["error"]))
    if not pilot_rows:
        lines += ["", "尚无可读的预实验状态记录。"]
    lines += ["", "## 正式预算与统计口径", ""]
    if plan:
        budget = plan["budget"]
        lines += [f"每轮训练 {budget['train_days_per_iteration']} 个完整日；验证 {budget['validation_days']} 日；测试 {budget['test_days']} 日；"
                  f"{budget['outer_iterations']} 轮交替采样与拟合；{budget['training_replicates']} 次独立训练。",
                  "训练、验证、测试及预实验种子分离。每轮只拟合当前冻结策略新产生的完整轨迹；验证净收益选模，测试不参与选择。",
                  "下表的 mean ± std 按测试场景计算。不同训练重复分别报告，不把同一测试场景上的两个训练重复当成两个独立场景。",
                  "按零终端预实验速度估计的全部计划耗时约 " + _fmt(plan.get("estimated_all_seconds_at_zero_pilot_speed"))
                  + " s，未计入训练及神经终端的额外开销；该估计不是截止前完成承诺。"]
    else:
        lines += ["正式预算尚未冻结；尚无正式完成结论。"]
    if report.get("phases"):
        lines += ["", "阶段状态：" + "；".join(name + "=" + str(value) for name,value in report["phases"].items()) + "。"]
    lines += ["", "## 已完成正式分组", ""]
    if not completed:
        lines += ["目前没有已完成的正式分组；CSV 仅含表头，不填入推测数值。"]
    else:
        lines += ["| 组别 | 训练重复 | 测试 n | 净收益/元 | 对零基线的配对净收益差/元 | 预约失败率/% | 随机服务率/% |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for name,group in sorted(completed.items()):
            stats = group.get("statistics", {})
            delta = group.get("paired_difference_from_zero", {}).get("net_profit_yuan")
            lines.append("| " + name + " | " + _fmt(group.get("metadata",{}).get("training_replicate"),0)
                + " | " + str(stats.get("net_profit_yuan",{}).get("n",len(group.get("scenario_seeds",[]))))
                + " | " + _summary(stats.get("net_profit_yuan")) + " | " + _summary(delta)
                + " | " + _summary(stats.get("reservation_failure_rate"),scale=100.)
                + " | " + _summary(stats.get("random_service_rate"),scale=100.) + " |")
        lines += ["", "完整均值、标准差、95% 区间及各指标有效 n 见两个正式结果 CSV；配对差值始终在相同场景内计算。基准配置复用已有结果时，其别名分组不代表新增独立样本。"]
    if skipped:
        lines += ["", "尚未完成、未计入统计的分组：" + "、".join(skipped) + "。"]
    if comparison:
        excluded = ", ".join("p" + str(value) for value in comparison.get("excluded_od_ids", []))
        lines += ["", "## O-D 输入版本修正", "",
                  f"原 {comparison.get('original_od_count')} 个 O-D 中固定剔除 {excluded}，新版本保留 {comparison.get('new_od_count')} 个有向 O-D。",
                  "原问题首段 e2(94 km)→s3(255 km) 长 161 km，超过 50% SOC 对应的 150 km。新固定集合已在 SOC=0.5 验证完整路径可达。",
                  "保持实际 SOC∈(0.5,1]、报告误差、六站资源及价格；重算新集合的预约权重和随机站点份额。逐车属性仍独立生成，没有按个体可达性重抽。",
                  "原 30 O-D 配置及原始场景保留；旧版失败或未完成轨迹不作为新版 MC 标签。详见 [配置对比](../configuration_comparison.json)。"]
    lines += ["", "## 与第 05 节原始草稿的口径对应", "",
              "实际实验在 24:00 停止，不继续处理期末已有请求，也不追加未完成预约的期末违约结算；期末未完成数与实际超时失败数分别报告。",
              "价值输入采用已确认的固定业务尺度，不从训练、验证或测试集拟合归一化统计。正式核心包含零、线性与完整 ReLU；05 的联合消融、库存消融和敏感性按计划另行完成。",
              "本次实现配置的求解后端为 COPT；原草稿环境表中的 Gurobi 占位不作为本次已执行环境事实。未完成分组不产生对应实验结论。", "",
              "## 已记录运行环境", "", "| 项目 | 记录 |", "|---|---|"]
    fields = [("操作系统",environment.get("operating_system")),("platform 原始字符串",environment.get("platform")),
              ("CPU 型号",environment.get("cpu_model")),("GPU 型号",environment.get("gpu_model")),
              ("GPU 显存/MiB",environment.get("gpu_memory_mib")),("训练设备协议",environment.get("training_device_protocol")),
              ("Python",environment.get("python")),
              ("执行程序",environment.get("executable")),("CPU 物理核",environment.get("cpu_physical")),
              ("CPU 逻辑核",environment.get("cpu_logical")),
              ("内存/GiB",environment["memory_bytes"] / 1024**3 if environment.get("memory_bytes") else None),
              ("每 worker 求解线程",environment.get("solver_threads_per_worker")),
              ("预实验进程并发数",environment.get("pilot_process_concurrency")),
              ("coptpy 软件包版本",environment.get("coptpy")),("PyTorch 软件包版本",environment.get("torch"))]
    lines += ["| " + name + " | " + _fmt(value,2) + " |" for name,value in fields]
    lines += ["", "环境来源为 execution_environment.json 和可选 hardware_environment.json。操作系统优先使用 WMI 的 Caption/Version，并保留 Python platform 原始字符串；硬件存在不表示已用其完成训练。软件包版本仅在报告生成器与 worker 使用同一 Python 路径时读取本地安装元数据；不将其冒充单独核验的原生求解器版本。缺失值以“—”表示。", "",
              "来源：[正式计划](../formal_suite/plan.json)、[正式机器可读结果](../formal_suite/report.json)、预实验 worker_status.json 与 journal/status.json。", ""]
    _atomic_text(paths["markdown"],"\n".join(lines))
    return {key:str(path) for key,path in paths.items()}
