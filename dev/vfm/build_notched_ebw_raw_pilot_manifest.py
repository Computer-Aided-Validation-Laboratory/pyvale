"""Build the targeted post-Round-1 raw-hybrid pilot manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    args = _parse_args()
    screen = args.screen.expanduser().resolve()
    output = args.output.expanduser().resolve()
    noise_model = args.noise_model.expanduser().resolve()
    raw_configs = {
        "raw_parsimonious": screen / "raw_parsimonious_objective.json",
        "raw_information_rich": screen / "raw_information_rich_objective.json",
    }
    for path in raw_configs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    cases = []
    for objective in (
        "current_29_57",
        "multiscale_equal_7_29_57",
        "raw_parsimonious",
        "raw_information_rich",
    ):
        for seed in args.seeds:
            for noise_scale in (0.0, 1.0):
                condition = "clean" if noise_scale == 0.0 else "noise1x"
                arguments = [
                    "--kernel-type", "bivariate_spd",
                    "--basis-growth-policy", "sensitivity_correction",
                    "--minimum-objective-improvement", "0",
                    "--max-basis-functions", str(args.max_basis_functions),
                    "--max-iterations", str(args.max_iterations),
                    "--max-evaluations", str(args.max_evaluations),
                    "--random-seed", str(seed),
                    "--stress-backend", args.stress_backend,
                ]
                if objective == "current_29_57":
                    arguments.extend(["--egi-windows", "29,57", "--egi-window-weights", "29,57"])
                elif objective == "multiscale_equal_7_29_57":
                    arguments.extend(["--egi-windows", "7,29,57", "--egi-window-weights", "1,1,1"])
                else:
                    config = json.loads(raw_configs[objective].read_text())
                    sizes = ",".join(str(int(window[0])) for window in config["egi_windows"])
                    arguments.extend([
                        "--objective-config", str(raw_configs[objective]),
                        "--egi-windows", sizes,
                        "--egi-window-weights", ",".join("1" for _ in config["egi_windows"]),
                    ])
                if noise_scale:
                    arguments.extend([
                        "--artificial-noise-model", str(noise_model),
                        "--artificial-noise-scale", "1",
                        "--artificial-noise-seed", str(args.noise_seed_base + seed),
                    ])
                cases.append({
                    "name": f"{objective}_{condition}_seed{seed:02d}",
                    "objective": objective,
                    "condition": condition,
                    "seed": seed,
                    "arguments": arguments,
                })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "purpose": "Targeted post-regime-repair raw hybrid pilot; projection excluded.",
        "case_count": len(cases),
        "cases": cases,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"pilot manifest={output} cases={len(cases)}", flush=True)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--noise-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=lambda value: tuple(int(item) for item in value.split(",")), default=(0, 1))
    parser.add_argument("--max-basis-functions", type=int, default=7)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--max-evaluations", type=int, default=15500)
    parser.add_argument("--stress-backend", choices=("numpy", "cython"), default="cython")
    parser.add_argument("--noise-seed-base", type=int, default=20260829)
    return parser.parse_args()


if __name__ == "__main__":
    main()
