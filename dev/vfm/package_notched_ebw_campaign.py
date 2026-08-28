"""Create a compact, relocatable archive of a gate campaign.

The default compact archive includes manifests, logs, analysis, result YAML,
and final parameter maps.  Derived final stress arrays are omitted because
they are large and can be reconstructed from the prepared experiment.  Use
--include-stress when a complete byte-for-byte result transfer is required.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import tarfile


def main() -> None:
    args = _parse_args()
    campaign = args.campaign_root.expanduser().resolve()
    if not (campaign / "campaign_manifest.json").is_file():
        raise FileNotFoundError(f"Campaign manifest not found below {campaign}")
    output = (
        args.output.expanduser().resolve()
        if args.output
        else campaign.parent / f"{campaign.name}_results_{datetime.now():%Y%m%d_%H%M}.tar.gz"
    )
    included = 0
    with tarfile.open(output, "w:gz") as archive:
        for path in sorted(campaign.rglob("*")):
            if not path.is_file() or not _include(path, campaign, args.include_stress):
                continue
            archive.add(path, arcname=Path(campaign.name) / path.relative_to(campaign))
            included += 1
    print(f"Created {output} with {included} files ({output.stat().st_size / 2**20:.1f} MiB)")


def _include(path: Path, campaign: Path, include_stress: bool) -> bool:
    relative = path.relative_to(campaign)
    if include_stress:
        return True
    if path.name == "final_identified_stress.npz":
        return False
    return (
        relative.parts[0] in {"analysis", "logs"}
        or path.name in {
            "campaign_manifest.json",
            "identification_result.yaml",
            "final_parameter_maps.npz",
        }
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-stress", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
