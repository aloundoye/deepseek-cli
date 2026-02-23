#!/usr/bin/env python3
"""Fail unless the last N parity nightly runs succeeded with passing journey reports."""

from __future__ import annotations

import io
import json
import os
import sys
import urllib.parse
import urllib.request
import zipfile


def _headers(token: str) -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "deepseek-parity-streak-gate",
    }


def _get_json(url: str, token: str) -> dict:
    req = urllib.request.Request(url, headers=_headers(token))
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _download_bytes(url: str, token: str) -> bytes:
    req = urllib.request.Request(url, headers=_headers(token))
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def _load_report_from_run(repo: str, run: dict, token: str) -> dict:
    artifacts = _get_json(run["artifacts_url"], token).get("artifacts", [])
    artifact = next((a for a in artifacts if a.get("name") == "parity-journey-report"), None)
    if artifact is None:
        raise RuntimeError(f"run {run.get('id')} missing parity-journey-report artifact")

    archive = _download_bytes(artifact["archive_download_url"], token)
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        names = zf.namelist()
        report_name = next((n for n in names if n.endswith("parity_journey_report.json")), None)
        if report_name is None:
            raise RuntimeError(
                f"run {run.get('id')} artifact missing parity_journey_report.json"
            )
        payload = zf.read(report_name).decode("utf-8")
        return json.loads(payload)


def _report_passes(report: dict) -> bool:
    if report.get("overall_pass") is False:
        return False
    required = report.get("required_journeys")
    if isinstance(required, dict):
        return all(bool(value) for value in required.values())
    return bool(report.get("overall_pass", True))


def main() -> int:
    repo = os.environ.get("GITHUB_REPOSITORY")
    token = os.environ.get("GITHUB_TOKEN")
    workflow_file = os.environ.get("PARITY_WORKFLOW_FILE", "parity-nightly.yml")
    required_streak = int(os.environ.get("PARITY_REQUIRED_STREAK", "3"))
    default_branch = os.environ.get("DEFAULT_BRANCH", "main")

    if not repo:
        print("GITHUB_REPOSITORY is required", file=sys.stderr)
        return 1
    if not token:
        print("GITHUB_TOKEN is required", file=sys.stderr)
        return 1

    encoded_branch = urllib.parse.quote(default_branch, safe="")
    runs_url = (
        f"https://api.github.com/repos/{repo}/actions/workflows/{workflow_file}/runs"
        f"?branch={encoded_branch}&status=completed&per_page=20"
    )
    payload = _get_json(runs_url, token)
    runs = payload.get("workflow_runs", [])

    if len(runs) < required_streak:
        print(
            f"Parity streak gate failed: need {required_streak} completed nightly runs, found {len(runs)}",
            file=sys.stderr,
        )
        return 1

    target_runs = runs[:required_streak]
    for idx, run in enumerate(target_runs, start=1):
        run_id = run.get("id")
        conclusion = run.get("conclusion")
        if conclusion != "success":
            print(
                f"Parity streak gate failed: run #{idx} (id={run_id}) conclusion={conclusion}",
                file=sys.stderr,
            )
            return 1

        try:
            report = _load_report_from_run(repo, run, token)
        except Exception as exc:  # noqa: BLE001
            print(f"Parity streak gate failed: {exc}", file=sys.stderr)
            return 1

        if not _report_passes(report):
            print(
                f"Parity streak gate failed: report in run id={run_id} did not pass required journeys",
                file=sys.stderr,
            )
            return 1

    print(
        f"Parity streak gate passed: last {required_streak} nightly runs succeeded with passing journey reports"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
