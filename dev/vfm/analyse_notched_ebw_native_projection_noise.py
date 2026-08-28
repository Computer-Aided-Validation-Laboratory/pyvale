"""Merge state shards from the native-DOF projection/noise campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from run_notched_ebw_native_projection_noise import (
    _load_rows, _report, _summaries, _write_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--noise-model", type=Path, required=True)
    parser.add_argument("--windows", default="7,15,29,57")
    parser.add_argument("--noise-scales", default="0,0.5,1,1.5")
    parser.add_argument("--noise-replicates", type=int, default=64)
    parser.add_argument("--expected-states", type=int, default=32)
    args = parser.parse_args()
    root = args.campaign_root.expanduser().resolve()
    paths = sorted(root.glob("state_*/projection_noise_rows.jsonl"))
    if not paths:
        raise RuntimeError(f"No state checkpoints found below {root}")
    rows = []
    for path in paths:
        rows.extend(_load_rows(path))
    state_count = len({row["state"] for row in rows})
    if state_count != args.expected_states:
        raise RuntimeError(
            f"Expected {args.expected_states} states, found {state_count}."
        )
    _write_csv(root / "projection_noise_scores.csv", rows)
    summary = _summaries(rows)
    _write_csv(root / "projection_noise_summary.csv", summary)
    noise_model = json.loads(args.noise_model.read_text(encoding="utf-8"))
    report_args = SimpleNamespace(noise_replicates=args.noise_replicates)
    report = _report(
        root, tuple(int(v) for v in args.windows.split(",")),
        tuple(float(v) for v in args.noise_scales.split(",")),
        report_args, noise_model, rows, summary,
    )
    (root / "campaign_summary.json").write_text(json.dumps({
        "states": state_count, "rows": len(rows), "shards": len(paths),
        "report": str(report),
    }, indent=2), encoding="utf-8")
    print(json.dumps({
        "states": state_count, "rows": len(rows), "shards": len(paths),
        "report": str(report),
    }, indent=2))


if __name__ == "__main__":
    main()
