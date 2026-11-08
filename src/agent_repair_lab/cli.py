"""Headless runner: python -m agent_repair_lab [--source ...] [--json] [--out FILE]

Exit codes: 0 accepted, 1 rejected by the verifier, 2 invalid proposal or repair agent failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .experiment import SOURCES, run_experiment


def _strip_traces(report: dict) -> dict:
    slim = json.loads(json.dumps(report))
    for variant in (slim.get("comparison") or {}).values():
        if variant:
            variant.pop("trace", None)
    return slim


def render_text(report: dict) -> str:
    lines = [
        f"{report['app']} {report['display_version']}  |  source: {report['source_label']}",
        f"Diagnosis case: {report['diagnosis_case']['id']} (budget {report['budget']} actions)",
        "",
        "Three-way comparison on the diagnosis case:",
    ]
    for name, title in (("baseline", "Baseline"), ("guarded", "Baseline + repeat guard"), ("edited", "Edited policy")):
        v = report["comparison"][name]
        if v is None:
            lines.append(f"  {title:<24} not run (no valid patch)")
            continue
        lines.append(
            f"  {title:<24} termination={v['termination']:<21} steps={v['steps']:<3} "
            f"loop={'yes' if v['loop_detected'] else 'no':<4} correct={'yes' if v['correct'] else 'no'}"
        )
    lines += ["", "Repair proposal:"]
    if report["repair_error"]:
        lines.append(f"  REPAIR AGENT FAILED: {report['repair_error']}")
    if report["proposal"] is not None:
        lines.append(f"  diagnosis: {report['proposal'].get('diagnosis')}")
    if report["validation"] and not report["validation"]["valid"]:
        lines.append("  validation FAILED (no weights written):")
        lines += [f"    - {e}" for e in report["validation"]["errors"]]
    for c in report["weight_changes"]:
        lines.append(f"  {c['label']} {c['parameter']}{c['index']}: {c['before']:.3f} -> {c['after']:.3f} (delta {c['delta']:+.3f})")
    if report["verification"]:
        lines += ["", "Independent verification:"]
        for c in report["verification"]["candidate"]:
            mark = "PASS" if c["passed"] else "FAIL"
            lines.append(f"  [{mark}] {c['role']:<17} {c['scenario_id']:<34} {c['failure'] or ''}")
        lines.append("")
        lines += [f"  - {r}" for r in report["verification"]["reasons"]]
    lines += [
        "",
        f"Original weights unchanged: {report['original_weights_unchanged']}",
        f"VERDICT: {report['status'].upper()}",
        "",
        report["credit"],
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent_repair_lab", description="Run the Agent Repair Lab experiment headlessly.")
    parser.add_argument("--source", choices=SOURCES, default="deterministic", help="proposal source (default: deterministic)")
    parser.add_argument("--json", action="store_true", help="print the JSON report instead of text")
    parser.add_argument("--full", action="store_true", help="include full traces in JSON output")
    parser.add_argument("--out", type=Path, help="also write the full JSON report to this file")
    args = parser.parse_args(argv)

    report = run_experiment(args.source)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(report if args.full else _strip_traces(report), indent=2))
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(render_text(report))
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
