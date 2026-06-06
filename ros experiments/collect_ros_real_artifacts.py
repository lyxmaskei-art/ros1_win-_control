from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

from analyze_ros_real_diagnostics import (
    DEFAULT_TAU,
    classify,
    finite,
    find_diagnostic_csv,
    load_rows,
    vector,
)
from compare_ros_real_runs import summarize_run, write_csv as write_compare_csv, write_markdown as write_compare_md


LIGHT_PATTERNS = (
    "run_config.json",
    "ros_real_preflight_report.json",
    "ros_real_cycle_diagnostics.csv",
    "ros_real_cycle_diagnostics_summary.json",
    "ros_real_diagnostic_analysis.json",
    "ros_real_diagnostic_analysis.md",
    "*_summary.json",
    "ros_real_step_time_summary.json",
    "ros_real_ab_comparison.csv",
    "ros_real_ab_comparison.md",
)

HEAVY_PATTERNS = (
    "ros_real_cycle_diagnostics.npz",
    "*_history.npz",
    "ros_real_step_time_history.npz",
    "*.png",
    "*.pdf",
    "*.svg",
)


def _safe_name(path):
    text = str(Path(path).resolve())
    for old, new in ((":", ""), ("\\", "_"), ("/", "_"), (" ", "_")):
        text = text.replace(old, new)
    return "".join(ch for ch in text if ch.isalnum() or ch in "._-")[-140:]


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _ensure_analysis(run_dir):
    csv_path = find_diagnostic_csv(run_dir)
    analysis_path = csv_path.with_name("ros_real_diagnostic_analysis.json")
    if analysis_path.exists():
        return analysis_path
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
    _write_json(analysis_path, payload)
    return analysis_path


def _copy_matches(src_dir, dst_dir, patterns):
    copied = []
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    for pattern in patterns:
        for src in sorted(src_dir.glob(pattern)):
            if not src.is_file():
                continue
            dst = dst_dir / src.name
            shutil.copy2(src, dst)
            copied.append(str(dst))
    return copied


def _zip_dir(src_dir, zip_path):
    src_dir = Path(src_dir)
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(src_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(src_dir))


def collect(paths, output_zip, include_heavy=False):
    run_dirs = [Path(path).resolve() for path in paths]
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "input_paths": [str(path) for path in run_dirs],
        "include_heavy": bool(include_heavy),
        "runs": [],
        "comparison": None,
        "warnings": [],
    }
    with tempfile.TemporaryDirectory(prefix="ros_real_triage_") as tmp:
        root = Path(tmp)
        for run_dir in run_dirs:
            run_bundle_dir = root / _safe_name(run_dir)
            run_bundle_dir.mkdir(parents=True, exist_ok=True)
            try:
                analysis_path = _ensure_analysis(run_dir)
                summary = summarize_run(run_dir)
            except Exception as exc:
                manifest["warnings"].append(
                    {
                        "path": str(run_dir),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                analysis_path = None
                summary = {}
            copied = _copy_matches(run_dir, run_bundle_dir, LIGHT_PATTERNS)
            if include_heavy:
                copied.extend(_copy_matches(run_dir, run_bundle_dir, HEAVY_PATTERNS))
            if analysis_path is not None and Path(analysis_path).parent != run_dir:
                shutil.copy2(analysis_path, run_bundle_dir / Path(analysis_path).name)
            manifest["runs"].append(
                {
                    "source": str(run_dir),
                    "bundle_dir": str(run_bundle_dir.relative_to(root)),
                    "analysis_json": "" if analysis_path is None else str(analysis_path),
                    "summary": summary,
                    "copied_file_count": len(copied),
                }
            )

        if len(run_dirs) > 1:
            rows = []
            for run_dir in run_dirs:
                try:
                    rows.append(summarize_run(run_dir))
                except Exception:
                    continue
            if rows:
                rows.sort(key=lambda row: float(row["score"]))
                compare_dir = root / "comparison"
                compare_dir.mkdir(parents=True, exist_ok=True)
                compare_csv = compare_dir / "ros_real_ab_comparison.csv"
                compare_md = compare_dir / "ros_real_ab_comparison.md"
                write_compare_csv(compare_csv, rows)
                write_compare_md(compare_md, rows)
                manifest["comparison"] = {
                    "csv": str(compare_csv.relative_to(root)),
                    "markdown": str(compare_md.relative_to(root)),
                    "rows": rows,
                }

        _write_json(root / "manifest.json", manifest)
        _zip_dir(root, output_zip)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Collect ROS real UR3e diagnostic artifacts into a shareable zip.")
    parser.add_argument("paths", nargs="+", help="Run directories to collect")
    parser.add_argument("--output-zip", default=None)
    parser.add_argument("--include-heavy", action="store_true", help="Also include history npz files and figures")
    args = parser.parse_args()

    output_zip = Path(args.output_zip) if args.output_zip else Path.cwd() / f"ros_real_triage_{time.strftime('%Y%m%d_%H%M%S')}.zip"
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    manifest = collect(args.paths, output_zip, include_heavy=args.include_heavy)
    print(json.dumps({"zip": str(output_zip), "manifest": manifest}, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
