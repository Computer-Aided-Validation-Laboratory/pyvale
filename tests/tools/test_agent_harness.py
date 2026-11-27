"""Subprocess regressions for the agent pytest runner."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


RUNNER = Path(__file__).resolve().parents[2] / "scripts" / "test_agent.py"


@pytest.mark.parametrize(
    ("source", "expected_code", "expected_diagnostic"),
    [
        ("def test_pass():\n    print('passing noise')\n", 0, "1 passed"),
        ("import pytest\ndef test_skip():\n    pytest.skip('optional')\n", 0,
         "1 skipped"),
        ("import pytest\n@pytest.mark.xfail\ndef test_xfail():\n    assert False\n",
         0, "1 xfailed"),
        ("def test_fail():\n    print('failure detail')\n    assert 1 == 2\n",
         1, "failure detail"),
        ("import pytest\n@pytest.fixture\ndef fixture():\n"
         "    raise RuntimeError('setup detail')\n"
         "def test_setup(fixture):\n    pass\n", 1, "setup detail"),
        ("import pytest\n@pytest.fixture\ndef fixture():\n    yield\n"
         "    raise RuntimeError('teardown detail')\n"
         "def test_teardown(fixture):\n    pass\n", 1, "teardown detail"),
        ("raise RuntimeError('collection detail')\n", 2, "collection detail"),
        ("", 5, "no tests"),
    ],
)
def test_agent_reports_pytest_outcomes(
    tmp_path: Path,
    source: str,
    expected_code: int,
    expected_diagnostic: str,
) -> None:
    test_path = tmp_path / "test_sample.py"
    test_path.write_text(source, encoding="utf-8")
    environment = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    completed = subprocess.run(
        [sys.executable, str(RUNNER), str(test_path)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == expected_code, completed.stdout + completed.stderr
    assert expected_diagnostic in completed.stdout
    assert "passing noise" not in completed.stdout
    assert completed.stderr == ""
    if expected_code == 0:
        assert len(completed.stdout.splitlines()) == 1
    else:
        log_line = next(
            line for line in completed.stdout.splitlines()
            if line.startswith("Full log: ")
        )
        log_path = Path(log_line.removeprefix("Full log: "))
        assert log_path.is_file()
        assert expected_diagnostic in log_path.read_text(encoding="utf-8")


def test_agent_discovers_new_suites_and_forwards_selectors(tmp_path: Path) -> None:
    suite_path = tmp_path / "tests" / "new_suite"
    suite_path.mkdir(parents=True)
    (suite_path / "test_first.py").write_text(
        "def test_selected():\n    pass\n", encoding="utf-8"
    )
    (suite_path / "second_test.py").write_text(
        "def test_other():\n    pass\n", encoding="utf-8"
    )
    environment = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    for arguments, expected in [([], "2 passed"), (["-k", "selected"], "1 passed")]:
        completed = subprocess.run(
            [sys.executable, str(RUNNER), *arguments],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert expected in completed.stdout


def test_agent_bounds_output_and_preserves_full_log(tmp_path: Path) -> None:
    test_path = tmp_path / "test_noisy.py"
    test_path.write_text(
        "import os\ndef test_noisy():\n"
        "    os.write(1, b'native noise\\n' * 1000)\n"
        "    print('x' * 100000)\n"
        "    assert False, 'assertion detail'\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--agent-max-lines", "20",
         "--agent-max-chars", "2000", str(test_path)],
        cwd=tmp_path,
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert len(completed.stdout.splitlines()) <= 20
    assert len(completed.stdout) <= 2000
    assert "assertion detail" in completed.stdout
    assert "truncated" in completed.stdout
    log_line = next(
        line for line in completed.stdout.splitlines() if line.startswith("Full log: ")
    )
    log_text = Path(log_line.removeprefix("Full log: ")).read_text(encoding="utf-8")
    assert "native noise\n" * 1000 in log_text
    assert "x" * 100000 in log_text


@pytest.mark.parametrize(
    ("arguments", "expected_code", "diagnostic"),
    [
        (["--unknown-pytest-option"], 4, "unrecognized arguments"),
        (["--help"], 0, "Pytest log tail:"),
        (["test_crash.py"], 17, "pytest did not finish"),
    ],
)
def test_agent_handles_runs_without_test_reports(
    tmp_path: Path,
    arguments: list[str],
    expected_code: int,
    diagnostic: str,
) -> None:
    (tmp_path / "test_crash.py").write_text(
        "import os\ndef test_crash():\n    os._exit(17)\n", encoding="utf-8"
    )
    completed = subprocess.run(
        [sys.executable, str(RUNNER), *arguments],
        cwd=tmp_path,
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == expected_code
    assert diagnostic in completed.stdout
