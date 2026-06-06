from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from analyze_ros_real_diagnostics import (
    DEFAULT_TAU,
    classify,
    finite,
    find_diagnostic_csv,
    load_rows,
    vector,
)


EXCLUDED_SUMMARY_NAMES = {
    "ros_real_cycle_diagnostics_summary.json",
    "ros_real_diagnostic_analysis.json",
    "ros_real_preflight_report.json",
}


def _load_json(path):
    with Path(path).open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _find_analysis_or_build(path):
    path = Path(path)
    if path.is_file() and path.name == "ros_real_diagnostic_analysis.json":
        return path, _load_json(path)
    csv_path = find_diagnostic_csv(path)
    analysis_path = csv_path.with_name("ros_real_diagnostic_analysis.json")
    if analysis_path.exists():
        return analysis_path, _load_json(analysis_path)
    rows = load_rows(csv_path)
    time_values = finite(vector(rows, "time_s"))
    tau = float((time_values[1:] - time_values[:-1]).mean()) if time_values.size > 1 else DEFAULT_TAU
    metrics, findings = classify(
        rows=rows,
        tau=tau,
        accel_warn_norm=12.0,
        accel_critical_norm=25.0,
        state_age_factor=2.0,
        period_factor=1.25,
    )
    payload = {
        "source_csv": str(csv_path),
        "metrics": metrics,
        "findings": findings,
    }
    analysis_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return analysis_path, payload


def _find_run_summary(path):
    path = Path(path)
    if path.is_file():
        path = path.parent
    for candidate in sorted(path.glob("*_summary.json")):
        if candidate.name not in EXCLUDED_SUMMARY_NAMES:
            try:
                data = _load_json(candidate)
            except Exception:
                continue
            if "ros_real_interface" in data or "mean_position_error_m" in data:
                return candidate, data
    return None, {}


def _count_severity(findings, severity):
    return sum(1 for item in findings if item.get("severity") == severity)


def _metric(metrics, key, stat, default=None):
    value = metrics.get(key, {})
    if isinstance(value, dict):
        return value.get(stat, default)
    return default


def _float_or_none(value):
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _infer_command_mode(path):
    text = str(path).lower()
    for mode in ("velocity_array", "joint_trajectory", "position_array"):
        if mode in text:
            return mode
    return "unknown"


def summarize_run(path):
    analysis_path, analysis = _find_analysis_or_build(path)
    run_dir = Path(analysis.get("source_csv", analysis_path)).parent
    summary_path, summary = _find_run_summary(run_dir)
    metrics = analysis.get("metrics", {})
    findings = analysis.get("findings", [])
    interface = summary.get("ros_real_interface", {})
    command_mode = interface.get("command_mode", summary.get("command_mode", _infer_command_mode(run_dir)))

    high_count = _count_severity(findings, "high")
    medium_count = _count_severity(findings, "medium")
    deadline = float(metrics.get("deadline_miss_ratio", 0.0) or 0.0)
    overrun = float(metrics.get("period_overrun_ratio", 0.0) or 0.0)
    tcp_pose_used_ratio = float(metrics.get("tcp_pose_used_ratio", 0.0) or 0.0)
    state_age_p95 = _metric(metrics, "state_age_s", "p95", 0.0) or 0.0
    accel_p95 = _metric(metrics, "command_accel_norm_rad_s2", "p95", 0.0) or 0.0
    tcp_fk_error_p95 = _metric(metrics, "tcp_fk_error_norm_m", "p95", 0.0) or 0.0
    position_error = _float_or_none(summary.get("mean_position_error_m"))
    drift_norm = _float_or_none(summary.get("final_joint_drift_norm_rad"))

    score = (
        1000.0 * high_count
        + 100.0 * medium_count
        + 500.0 * deadline
        + 300.0 * overrun
        + 10.0 * state_age_p95
        + 0.1 * accel_p95
        + 20.0 * tcp_fk_error_p95
    )
    if position_error is not None:
        score += 100.0 * position_error
    if drift_norm is not None:
        score += 10.0 * drift_norm

    top_finding = findings[0]["title"] if findings else "none"
    return {
        "run_dir": str(run_dir),
        "analysis_json": str(analysis_path),
        "summary_json": "" if summary_path is None else str(summary_path),
        "command_mode": command_mode,
        "command_topic": interface.get("command_topic", summary.get("command_topic", "")),
        "trajectory_name": summary.get("trajectory_name", ""),
        "method": summary.get("solver_type", summary.get("method", "")),
        "score": score,
        "high_findings": high_count,
        "medium_findings": medium_count,
        "top_finding": top_finding,
        "deadline_miss_ratio": deadline,
        "period_overrun_ratio": overrun,
        "tcp_pose_used_ratio": tcp_pose_used_ratio,
        "cycle_period_p99_s": _metric(metrics, "cycle_period_s", "p99", ""),
        "state_age_p95_s": state_age_p95,
        "command_accel_p95_rad_s2": accel_p95,
        "tcp_fk_error_p95_m": tcp_fk_error_p95,
        "mean_position_error_m": "" if position_error is None else position_error,
        "final_joint_drift_norm_rad": "" if drift_norm is None else drift_norm,
    }


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows):
    lines = [
        "# ROS real A/B comparison",
        "",
        "| Rank | Command mode | Score | High | Medium | Deadline miss | Period overrun | TCP used | State age p95 (s) | Command accel p95 | TCP-FK p95 (m) | Mean error (m) | Final drift (rad) | Top finding |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for idx, row in enumerate(rows, start=1):
        lines.append(
            "| {rank} | {mode} | {score:.6g} | {high} | {medium} | {deadline:.3%} | {overrun:.3%} | {tcp_used:.3%} | {age:.6g} | {accel:.6g} | {tcp_fk:.6g} | {error} | {drift} | {finding} |".format(
                rank=idx,
                mode=row["command_mode"],
                score=float(row["score"]),
                high=row["high_findings"],
                medium=row["medium_findings"],
                deadline=float(row["deadline_miss_ratio"]),
                overrun=float(row["period_overrun_ratio"]),
                tcp_used=float(row["tcp_pose_used_ratio"]),
                age=float(row["state_age_p95_s"]),
                accel=float(row["command_accel_p95_rad_s2"]),
                tcp_fk=float(row["tcp_fk_error_p95_m"]),
                error=row["mean_position_error_m"],
                drift=row["final_joint_drift_norm_rad"],
                finding=row["top_finding"],
            )
        )
    best = rows[0]
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"Best diagnostic score: `{best['command_mode']}` from `{best['run_dir']}`.",
        ]
    )
    if int(best["high_findings"]) > 0:
        lines.append("The best run still has high-severity findings; fix those before treating interface choice as settled.")
    else:
        lines.append("No high-severity finding in the best run. If visual shake also improved, keep this command mode for the next tuning pass.")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Compare multiple ROS real diagnostic runs.")
    parser.add_argument("paths", nargs="+", help="Run directories or ros_real_diagnostic_analysis.json files")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    rows = [summarize_run(path) for path in args.paths]
    rows.sort(key=lambda row: float(row["score"]))
    out_dir = Path(args.output_dir) if args.output_dir else Path(rows[0]["run_dir"]).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ros_real_ab_comparison.csv"
    md_path = out_dir / "ros_real_ab_comparison.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    print(json.dumps({"rows": rows, "csv": str(csv_path), "markdown": str(md_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
