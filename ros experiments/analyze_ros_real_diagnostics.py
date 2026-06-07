from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DIAGNOSTIC_CSV_NAME = "ros_real_cycle_diagnostics.csv"
DEFAULT_TAU = 0.005
ANALYSIS_SCHEMA_VERSION = 2


def _to_float(value, default=float("nan")):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def find_diagnostic_csv(path):
    path = Path(path)
    if path.is_file():
        return path
    candidates = sorted(path.rglob(DIAGNOSTIC_CSV_NAME), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"Cannot find {DIAGNOSTIC_CSV_NAME} under {path}")
    return candidates[0]


def load_rows(csv_path):
    rows = []
    with Path(csv_path).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    return rows


def vector(rows, key):
    return np.asarray([_to_float(row.get(key)) for row in rows], dtype=float)


def finite(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def stats(values):
    values = finite(values)
    if values.size == 0:
        return {"mean": None, "median": None, "p95": None, "p99": None, "max": None}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def ratio(flags):
    flags = list(flags)
    if not flags:
        return 0.0
    return float(sum(1 for flag in flags if flag)) / float(len(flags))


def add_finding(findings, severity, title, evidence, action):
    findings.append(
        {
            "severity": severity,
            "title": title,
            "evidence": evidence,
            "action": action,
        }
    )


def classify(rows, tau, accel_warn_norm, accel_critical_norm, state_age_factor, period_factor):
    metrics = {
        "samples": len(rows),
        "tau_s": float(tau),
        "cycle_period_s": stats(vector(rows, "cycle_period_s")),
        "cycle_work_s": stats(vector(rows, "cycle_work_s")),
        "state_read_s": stats(vector(rows, "state_read_s")),
        "state_age_s": stats(vector(rows, "state_age_s")),
        "controller_step_s": stats(vector(rows, "controller_step_s")),
        "publish_s": stats(vector(rows, "publish_s")),
        "sleep_s": stats(vector(rows, "sleep_s")),
        "command_step_norm_rad": stats(vector(rows, "command_step_norm_rad")),
        "command_step_max_abs_rad": stats(vector(rows, "command_step_max_abs_rad")),
        "command_velocity_norm_rad_s": stats(vector(rows, "command_velocity_norm_rad_s")),
        "command_accel_norm_rad_s2": stats(vector(rows, "command_accel_norm_rad_s2")),
        "command_jerk_norm_rad_s3": stats(vector(rows, "command_jerk_norm_rad_s3")),
        "raw_command_velocity_norm_rad_s": stats(vector(rows, "raw_command_velocity_norm_rad_s")),
        "feedback_velocity_norm_rad_s": stats(vector(rows, "feedback_velocity_norm_rad_s")),
        "tcp_pose_age_s": stats(vector(rows, "tcp_pose_age_s")),
        "tcp_fk_error_norm_m": stats(vector(rows, "tcp_fk_error_norm_m")),
        "deadline_miss_ratio": ratio(_to_bool(row.get("deadline_miss")) for row in rows),
        "period_overrun_ratio": ratio(_to_bool(row.get("period_overrun")) for row in rows),
        "tcp_pose_used_ratio": ratio(_to_bool(row.get("tcp_pose_used")) for row in rows),
    }

    findings = []
    period_p99 = metrics["cycle_period_s"]["p99"]
    work_p99 = metrics["cycle_work_s"]["p99"]
    state_age_p95 = metrics["state_age_s"]["p95"]
    accel_p95 = metrics["command_accel_norm_rad_s2"]["p95"]
    accel_max = metrics["command_accel_norm_rad_s2"]["max"]
    jerk_p95 = metrics["command_jerk_norm_rad_s3"]["p95"]
    command_velocity_p95 = metrics["command_velocity_norm_rad_s"]["p95"]
    feedback_velocity_p95 = metrics["feedback_velocity_norm_rad_s"]["p95"]
    step_max = metrics["command_step_max_abs_rad"]["max"]
    controller_p99 = metrics["controller_step_s"]["p99"]
    publish_p99 = metrics["publish_s"]["p99"]
    tcp_pose_age_p95 = metrics["tcp_pose_age_s"]["p95"]
    tcp_fk_error_p95 = metrics["tcp_fk_error_norm_m"]["p95"]

    if metrics["deadline_miss_ratio"] > 0.01 or (work_p99 is not None and work_p99 > tau):
        add_finding(
            findings,
            "high",
            "Whole-loop deadline misses",
            f"cycle_work_s p99={work_p99:.6g}s, deadline_miss_ratio={metrics['deadline_miss_ratio']:.3%}",
            "Lower control rate to tau=0.008 or 0.01 for a test, reduce plotting/logging during live runs, and inspect controller_step_s versus publish_s.",
        )

    if metrics["period_overrun_ratio"] > 0.01 or (period_p99 is not None and period_p99 > period_factor * tau):
        add_finding(
            findings,
            "high",
            "ROS loop period jitter",
            f"cycle_period_s p99={period_p99:.6g}s, period_overrun_ratio={metrics['period_overrun_ratio']:.3%}",
            "Run on a less loaded Ubuntu session, avoid other ROS nodes consuming CPU, and compare tau=0.005 with tau=0.01.",
        )

    if state_age_p95 is not None and state_age_p95 > state_age_factor * tau:
        add_finding(
            findings,
            "high",
            "Stale joint-state feedback",
            f"state_age_s p95={state_age_p95:.6g}s, threshold={state_age_factor * tau:.6g}s",
            "Increase joint-state publish rate or reduce outer-loop rate. Do not tune gains until feedback freshness is fixed.",
        )

    if accel_p95 is not None and accel_p95 > accel_critical_norm:
        add_finding(
            findings,
            "high",
            "Command acceleration spikes",
            f"command_accel_norm_rad_s2 p95={accel_p95:.6g}, max={accel_max:.6g}",
            "Enable or lower --max-command-accel, reduce --theta-dot-limit, and reduce solver/task gains before increasing trajectory speed.",
        )
    elif accel_p95 is not None and accel_p95 > accel_warn_norm:
        add_finding(
            findings,
            "medium",
            "Command acceleration is high",
            f"command_accel_norm_rad_s2 p95={accel_p95:.6g}, max={accel_max:.6g}",
            "Keep --max-command-accel enabled and sweep 6, 8, 12 rad/s^2 before changing the main gains.",
        )

    if (
        command_velocity_p95 is not None
        and feedback_velocity_p95 is not None
        and command_velocity_p95 > 0.2
        and feedback_velocity_p95 > 0.0
        and command_velocity_p95 / feedback_velocity_p95 > 4.0
    ):
        add_finding(
            findings,
            "high",
            "Command velocity greatly exceeds feedback velocity",
            (
                f"command_velocity_norm_rad_s p95={command_velocity_p95:.6g}, "
                f"feedback_velocity_norm_rad_s p95={feedback_velocity_p95:.6g}, "
                f"ratio={command_velocity_p95 / feedback_velocity_p95:.3g}"
            ),
            (
                "The outer controller is producing velocity-like increments much faster than the real controller follows. "
                "Prefer velocity_array with accel/jerk limits, or use a multi-point joint_trajectory window instead of dense single-step position targets."
            ),
        )

    if jerk_p95 is not None and jerk_p95 > 200.0:
        add_finding(
            findings,
            "medium",
            "Command jerk is high",
            f"command_jerk_norm_rad_s3 p95={jerk_p95:.6g}",
            "Enable or lower --max-command-jerk, especially for velocity_array tests where acceleration spikes showed visible shake.",
        )

    if step_max is not None and step_max > 0.01:
        add_finding(
            findings,
            "medium",
            "Large per-cycle position target steps",
            f"command_step_max_abs_rad max={step_max:.6g}",
            "Use a smaller --theta-dot-limit or lower --max-command-step. For tau=0.005 and theta-dot-limit=0.4, expect about 0.002 rad per joint per cycle.",
        )

    if controller_p99 is not None and controller_p99 > 0.5 * tau:
        add_finding(
            findings,
            "medium",
            "Controller compute time consumes much of the cycle",
            f"controller_step_s p99={controller_p99:.6g}s",
            "Reduce inner steps, simplify the live controller path, or test a slower outer loop before blaming the robot servo layer.",
        )

    if publish_p99 is not None and publish_p99 > 0.25 * tau:
        add_finding(
            findings,
            "medium",
            "ROS publish path is not negligible",
            f"publish_s p99={publish_p99:.6g}s",
            "Check ROS networking and controller subscriber load; publish latency can make high-rate position streaming unstable.",
        )

    if tcp_pose_age_p95 is not None and tcp_pose_age_p95 > state_age_factor * tau:
        add_finding(
            findings,
            "medium",
            "Measured TCP pose is stale",
            f"tcp_pose_age_s p95={tcp_pose_age_p95:.6g}s, threshold={state_age_factor * tau:.6g}s",
            "Use TCP pose mainly as offline evidence unless its publish rate and timestamp age are close to the control loop.",
        )

    if metrics["tcp_pose_used_ratio"] > 0.0 and metrics["tcp_pose_used_ratio"] < 0.95:
        add_finding(
            findings,
            "medium",
            "Measured TCP pose is intermittent",
            f"tcp_pose_used_ratio={metrics['tcp_pose_used_ratio']:.3%}",
            "Increase TCP pose publish rate or leave --tcp-pose-topic empty so tracking metrics are not a mix of measured TCP and local FK.",
        )

    if tcp_fk_error_p95 is not None and tcp_fk_error_p95 > 0.01:
        add_finding(
            findings,
            "medium",
            "Measured TCP and local FK disagree",
            f"tcp_fk_error_norm_m p95={tcp_fk_error_p95:.6g}m",
            "Check TCP offset, base/tool frames, and whether the measured TCP topic is in the same frame as the local DH model.",
        )

    if not findings:
        add_finding(
            findings,
            "low",
            "No obvious timing or command-smoothness fault in diagnostics",
            "All checked p95/p99 values are within the default thresholds.",
            "If the robot still visibly shakes, suspect the position-command interface or TCP/model mismatch; test velocity or time-stamped trajectory control.",
        )

    severity_rank = {"high": 0, "medium": 1, "low": 2}
    findings.sort(key=lambda item: severity_rank.get(item["severity"], 99))
    return metrics, findings


def write_markdown(path, csv_path, metrics, findings):
    lines = [
        "# ROS real diagnostics analysis",
        "",
        f"Source: `{csv_path}`",
        "",
        "## Key metrics",
        "",
        f"- Samples: {metrics['samples']}",
        f"- tau: {metrics['tau_s']:.6g} s",
        f"- deadline miss ratio: {metrics['deadline_miss_ratio']:.3%}",
        f"- period overrun ratio: {metrics['period_overrun_ratio']:.3%}",
        f"- measured TCP pose used ratio: {metrics['tcp_pose_used_ratio']:.3%}",
    ]
    for key in (
        "cycle_period_s",
        "cycle_work_s",
        "state_age_s",
        "controller_step_s",
        "publish_s",
        "command_step_max_abs_rad",
        "command_velocity_norm_rad_s",
        "command_accel_norm_rad_s2",
        "command_jerk_norm_rad_s3",
        "raw_command_velocity_norm_rad_s",
        "feedback_velocity_norm_rad_s",
        "tcp_pose_age_s",
        "tcp_fk_error_norm_m",
    ):
        item = metrics[key]
        lines.append(
            f"- {key}: median={item['median']}, p95={item['p95']}, p99={item['p99']}, max={item['max']}"
        )
    lines.extend(["", "## Findings", ""])
    for idx, finding in enumerate(findings, start=1):
        lines.extend(
            [
                f"{idx}. [{finding['severity'].upper()}] {finding['title']}",
                f"   Evidence: {finding['evidence']}",
                f"   Action: {finding['action']}",
            ]
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Analyze ROS real UR3e cycle diagnostics.")
    parser.add_argument("path", help="Run directory or ros_real_cycle_diagnostics.csv")
    parser.add_argument("--tau", type=float, default=None, help="Override control period in seconds")
    parser.add_argument("--accel-warn-norm", type=float, default=12.0)
    parser.add_argument("--accel-critical-norm", type=float, default=25.0)
    parser.add_argument("--state-age-factor", type=float, default=2.0)
    parser.add_argument("--period-factor", type=float, default=1.25)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    csv_path = find_diagnostic_csv(args.path)
    rows = load_rows(csv_path)
    inferred_tau = args.tau
    if inferred_tau is None:
        time_values = finite(vector(rows, "time_s"))
        inferred_tau = float(np.median(np.diff(time_values))) if time_values.size > 1 else DEFAULT_TAU
    metrics, findings = classify(
        rows=rows,
        tau=inferred_tau,
        accel_warn_norm=args.accel_warn_norm,
        accel_critical_norm=args.accel_critical_norm,
        state_age_factor=args.state_age_factor,
        period_factor=args.period_factor,
    )
    result = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "source_csv": str(csv_path),
        "metrics": metrics,
        "findings": findings,
    }

    output_json = Path(args.output_json) if args.output_json else csv_path.with_name("ros_real_diagnostic_analysis.json")
    output_md = Path(args.output_md) if args.output_md else csv_path.with_name("ros_real_diagnostic_analysis.md")
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(output_md, csv_path, metrics, findings)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")


if __name__ == "__main__":
    main()
