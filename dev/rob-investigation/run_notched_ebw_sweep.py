"""Run a resumable notched-EBW identification sweep from one JSON manifest."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CALLER = Path(__file__).parents[1] / "vfm" / "call_notched_ebw_bivariate_identification.py"


def main() -> None:
    args = _parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    # Preserve the transfer-bundle parent even when a local preflight uses a
    # symlink for the data directory; ``resolve()`` would lose the marker path.
    data_dir = args.data_dir.absolute()
    experiment_file = data_dir / "experiment_data.yaml"
    _validate_transfer(data_dir)
    _validate_manifest(manifest)
    _validate_runtime(experiment_file, args.output_root, manifest)

    if args.preflight_only:
        print(json.dumps({"status": "preflight_ok", "cases": len(manifest["cases"]), "input": str(experiment_file)}, indent=2))
        return

    run_id = str(manifest["run_id"])
    run_dir = args.output_root / run_id
    logs_dir = run_dir / "logs"
    cases_root = run_dir / "cases"
    logs_dir.mkdir(parents=True, exist_ok=True)
    cases_root.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "batch_started.json", _event("running", run_id=run_id))

    common = dict(manifest.get("common", {}))
    failures: list[str] = []
    for index, case in enumerate(manifest["cases"], start=1):
        case_id = str(case["id"])
        result_file = cases_root / case_id / "identification_result.yaml"
        if result_file.exists() and not args.rerun_completed:
            print(f"[{index}/{len(manifest['cases'])}] skip completed {case_id}", flush=True)
            continue

        options = {**common, **case.get("options", {})}
        command = _build_command(experiment_file, cases_root, case_id, options)
        _write_json(logs_dir / f"{case_id}.command.json", {"command": command, "options": options})
        print(f"[{index}/{len(manifest['cases'])}] start {case_id}", flush=True)
        started = datetime.now(timezone.utc)
        with (logs_dir / f"{case_id}.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
        _write_json(
            logs_dir / f"{case_id}.status.json",
            {
                **_event("complete" if completed.returncode == 0 else "failed", case_id=case_id),
                "return_code": completed.returncode,
                "elapsed_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
            },
        )
        if completed.returncode != 0:
            failures.append(case_id)
            print(f"[{index}/{len(manifest['cases'])}] FAILED {case_id}; continuing", flush=True)
        else:
            print(f"[{index}/{len(manifest['cases'])}] complete {case_id}", flush=True)

    final_status = "complete" if not failures else "complete_with_failures"
    _write_json(run_dir / "batch_finished.json", _event(final_status, run_id=run_id, failed_cases=failures))
    if failures:
        raise SystemExit(f"Batch completed with failed cases: {', '.join(failures)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--rerun-completed", action="store_true")
    return parser.parse_args()


def _validate_transfer(data_dir: Path) -> None:
    marker = data_dir.parent / "TRANSFER_COMPLETE.json"
    size_manifest = data_dir.parent / "manifest.tsv"
    if not marker.is_file():
        raise FileNotFoundError(f"Missing transfer marker: {marker}")
    if not size_manifest.is_file():
        raise FileNotFoundError(f"Missing file-size manifest: {size_manifest}")
    marker_data = json.loads(marker.read_text(encoding="utf-8"))
    if marker_data.get("status") != "complete":
        raise ValueError(f"Transfer marker does not report complete status: {marker}")
    with size_manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not rows or set(rows[0]) != {"relative_path", "size_bytes"}:
        raise ValueError(f"Invalid transfer manifest columns: {size_manifest}")
    for row in rows:
        path = data_dir.parent / row["relative_path"]
        expected = int(row["size_bytes"])
        if not path.is_file() or path.stat().st_size != expected:
            raise ValueError(f"Transferred file missing or wrong size: {path}")


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if not manifest.get("run_id") or not isinstance(manifest.get("cases"), list) or not manifest["cases"]:
        raise ValueError("Sweep manifest requires non-empty run_id and cases.")
    ids = [str(case.get("id", "")) for case in manifest["cases"]]
    if any(not case_id or not case_id.replace("-", "").replace("_", "").isalnum() for case_id in ids):
        raise ValueError("Case IDs may contain only letters, digits, hyphens and underscores.")
    if len(ids) != len(set(ids)):
        raise ValueError("Sweep case IDs must be unique.")


def _validate_runtime(experiment_file: Path, output_root: Path, manifest: dict[str, Any]) -> None:
    required = (experiment_file, experiment_file.parent / "known_parameter_maps.npz")
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    output_root.mkdir(parents=True, exist_ok=True)
    probe = output_root / ".pyvale_sweep_write_test"
    probe.write_text("ok\n", encoding="utf-8")
    probe.unlink()
    if not CALLER.is_file():
        raise FileNotFoundError(CALLER)
    subprocess.run([sys.executable, str(CALLER), "--help"], check=True, stdout=subprocess.DEVNULL)
    if manifest.get("common", {}).get("stress_backend") == "cython":
        subprocess.run(
            [sys.executable, "-c", "from cython_stress_recon.pyvale_adapter import CompiledLinearHardeningLaw"],
            check=True,
            stdout=subprocess.DEVNULL,
        )


def _build_command(input_file: Path, output_root: Path, case_id: str, options: dict[str, Any]) -> list[str]:
    command = [sys.executable, str(CALLER), "--input", str(input_file), "--output-root", str(output_root), "--run-name", case_id, "--no-progress"]
    for name, value in options.items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(flag)
        elif isinstance(value, list):
            command.extend((flag, ",".join(str(item) for item in value)))
        else:
            command.extend((flag, str(value)))
    return command


def _event(status: str, **extra: Any) -> dict[str, Any]:
    return {"status": status, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "host": os.environ.get("COMPUTERNAME", os.uname().nodename if hasattr(os, "uname") else "unknown"), **extra}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
