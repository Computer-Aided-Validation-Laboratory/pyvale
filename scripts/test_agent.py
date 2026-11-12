#!/usr/bin/env python3
"""Run pytest with bounded console diagnostics and a complete local log."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


@dataclass(slots=True)
class AgentReporter:
    report_path: Path

    def write_event(self, event: dict[str, str | int]) -> None:
        with self.report_path.open("a", encoding="utf-8") as report_file:
            report_file.write(json.dumps(event) + "\n")

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.failed:
            captured = "\n".join(
                f"{name}\n{content}" for name, content in report.sections
            )
            self.write_event({
                "kind": "failure",
                "id": report.nodeid,
                "phase": report.when,
                "diagnostic": report.longreprtext,
                "captured": captured,
            })

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if report.failed:
            self.write_event({
                "kind": "failure",
                "id": report.nodeid,
                "phase": "collection",
                "diagnostic": report.longreprtext,
                "captured": "",
            })

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        terminal = session.config.pluginmanager.get_plugin("terminalreporter")
        counts = []
        if terminal is not None:
            for outcome, reports in terminal.stats.items():
                if outcome and reports:
                    counts.append(f"{len(reports)} {outcome}")
        self.write_event({
            "kind": "summary",
            "diagnostic": ", ".join(counts) or "no tests completed",
            "exit_code": int(exitstatus),
        })


def run_pytest_child(arguments: list[str]) -> int:
    import pytest

    report_path = Path(arguments[0])
    return int(pytest.main(
        ["-q", "--tb=short", "--color=no", *arguments[1:]],
        plugins=[AgentReporter(report_path)],
    ))


def run_agent_tests(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--agent-max-lines", type=int, default=200)
    parser.add_argument("--agent-max-chars", type=int, default=24000)
    options, pytest_arguments = parser.parse_known_args(arguments)
    if options.agent_max_lines < 5 or options.agent_max_chars < 512:
        parser.error("agent output limits must be at least 5 lines and 512 characters")

    log_directory = Path(tempfile.mkdtemp(prefix="pyvale-tests-"))
    log_path = log_directory / "pytest.log"
    report_path = log_directory / "reports.jsonl"
    report_path.touch()
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_pytest-child",
        str(report_path),
        *(pytest_arguments or ["tests/"]),
    ]
    start_time = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
        try:
            exit_code = process.wait()
        except KeyboardInterrupt:
            # The child shares our process group and also receives terminal SIGINT.
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait()
            exit_code = 2

    events: list[dict[str, str | int]] = []
    if report_path.exists():
        with report_path.open(encoding="utf-8") as report_file:
            for line in report_file:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    # A killed worker can leave its final record incomplete.
                    break
    failures = [event for event in events if event["kind"] == "failure"]
    summaries = [event for event in events if event["kind"] == "summary"]
    summary = str(summaries[-1]["diagnostic"]) if summaries else "pytest did not finish"
    status = "PASS" if exit_code == 0 else "FAIL"
    elapsed = time.monotonic() - start_time
    header = f"{status}: {summary} ({elapsed:.1f}s; exit {exit_code})"
    footer = (
        [f"Full log: {log_path}", f"Reports: {report_path}"]
        if exit_code != 0 else []
    )
    truncation_notice = "... diagnostics truncated; inspect the saved artifacts."
    print(header)

    # Reserve space for the summary, truncation notice, and artifact paths.
    reserved_lines = [header, *footer, truncation_notice]
    remaining_lines = options.agent_max_lines - len(reserved_lines)
    remaining_chars = max(
        0,
        options.agent_max_chars - sum(len(line) + 1 for line in reserved_lines),
    )
    truncated = False

    def print_bounded(diagnostic: str) -> None:
        nonlocal remaining_lines, remaining_chars, truncated
        for line in diagnostic.splitlines():
            if remaining_lines <= 0 or remaining_chars <= 0:
                truncated = True
                return
            clipped = line[:max(0, remaining_chars - 1)]
            print(clipped)
            remaining_lines -= 1
            remaining_chars -= len(clipped) + 1
            if clipped != line:
                truncated = True
                return

    # Prioritise failure identities and assertions over captured output.
    for failure in failures:
        print_bounded(f"\n{failure['id']} [{failure['phase']}]")
    for failure in failures:
        print_bounded(str(failure["diagnostic"]))
    for failure in failures:
        if failure["captured"]:
            print_bounded(f"\n{failure['id']}: {failure['captured']}")
    if not summaries or (exit_code != 0 and not failures):
        with log_path.open(encoding="utf-8", errors="replace") as log_file:
            log_tail = deque(log_file, maxlen=options.agent_max_lines)
        print_bounded("\nPytest log tail:\n" + "".join(log_tail))
    if truncated:
        print(truncation_notice)
    for line in footer:
        print(line)
    return exit_code if exit_code >= 0 else 128 - exit_code


if __name__ == "__main__":
    if sys.argv[1:2] == ["--_pytest-child"]:
        sys.exit(run_pytest_child(sys.argv[2:]))
    sys.exit(run_agent_tests(sys.argv[1:]))
