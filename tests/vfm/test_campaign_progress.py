import json
from pathlib import Path

from pyvale.vfm.campaignprogress import ProgressEstimate, atomic_write_json


def test_progress_estimate_has_eta_after_first_completion(monkeypatch):
    monkeypatch.setattr("pyvale.vfm.campaignprogress.time.monotonic", lambda: 20.0)
    estimate = ProgressEstimate.from_counts(2, 10, 0.0)
    assert estimate.eta_seconds == 80.0
    assert "complete=2/10" in estimate.line()


def test_atomic_manifest_replaces_file(tmp_path: Path):
    path = tmp_path / "manifest.json"
    atomic_write_json(path, {"cases": [{"status": "pending"}]})
    payload = json.loads(path.read_text())
    assert payload["cases"][0]["status"] == "pending"
    assert "updated_at" in payload
    assert not path.with_suffix(".json.tmp").exists()
