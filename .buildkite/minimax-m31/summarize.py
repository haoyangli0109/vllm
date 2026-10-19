# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Print a Markdown summary of the mqa results for the Buildkite annotation.

Exits 1 when a metric in thresholds.yaml is missing or below its minimum.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml


def load_results(mqa_dir: Path) -> list[dict]:
    results = []
    for path in sorted(mqa_dir.rglob("result_*.json")):
        if path.name.endswith("_server_info.json"):
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and "task" in data and "scores" in data:
            results.append(data)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mqa-dir", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--vigil-status", type=int, required=True)
    args = parser.parse_args()

    thresholds = yaml.safe_load(args.thresholds.read_text()) or {}
    results = load_results(args.mqa_dir)
    scores = {r["task"]: r["scores"] for r in results}
    durations = {r["task"]: r.get("eval_duration_s") for r in results}

    below = []
    for task, minimums in thresholds.items():
        for metric, minimum in minimums.items():
            value = scores.get(task, {}).get(metric)
            if not isinstance(value, (int, float)) or value < minimum:
                below.append((task, metric, value, minimum))

    status = (
        "passed" if args.vigil_status == 0 else f"failed (exit {args.vigil_status})"
    )
    print("### MiniMax-M3.1 TP4 on B300\n")
    print(f"- nho2 commit: `{args.commit}`")
    print(f"- Image: `{args.image}`")
    print(f"- vigil: **{status}**")
    print(f"- Score thresholds: **{'failed' if below else 'passed'}**\n")

    if results:
        print("| Task | Metric | Score | Minimum | Eval time |")
        print("| --- | --- | --- | --- | --- |")
        for task, task_scores in scores.items():
            duration = durations[task]
            duration = (
                f"{duration / 60:.0f} min"
                if isinstance(duration, (int, float))
                else "?"
            )
            for metric, value in sorted(task_scores.items()):
                score = f"{value:.4f}" if isinstance(value, float) else value
                minimum = thresholds.get(task, {}).get(metric)
                if minimum is None:
                    check = ""
                else:
                    failed = (task, metric, value, minimum) in below
                    check = f"{minimum} {'❌' if failed else '✅'}"
                print(f"| {task} | {metric} | **{score}** | {check} | {duration} |")
    else:
        print("No mqa results were produced.")

    for task, metric, value, minimum in below:
        found = "missing" if value is None else f"= {value}"
        print(f"\n❌ `{task}` `{metric}` {found}, minimum {minimum}")
    sys.exit(1 if below else 0)


if __name__ == "__main__":
    main()
