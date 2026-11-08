"""Experiment runner shared by the Streamlit UI, the CLI and the tests."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone

from . import CREDIT, DISPLAY_VERSION, __version__
from .environment import Scenario, load_diagnosis_case, load_verification_cases
from .llm_adapter import OllamaRepairAgent
from .patch import PatchConstraints, apply_edits, validate_proposal, weight_diff
from .policy import MLPPolicy, load_baseline_policy
from .repair_agent import (
    DeterministicRepairAgent,
    NegativeControlAgent,
    RepairAgent,
    RepairUnavailable,
    build_repair_request,
)
from .task_agent import DEFAULT_ACTION_BUDGET, RepeatGuard, TaskAgent, longest_repeat
from .verifier import Verifier, evaluate_run

ACCEPTED, REJECTED, INVALID, REPAIR_FAILED = "accepted", "rejected", "invalid_proposal", "repair_failed"
EXIT_CODES = {ACCEPTED: 0, REJECTED: 1, INVALID: 2, REPAIR_FAILED: 2}

SOURCES = ("deterministic", "llm", "negative:force-return", "negative:force-fetch", "negative:rebalance", "negative:excessive")


def make_repair_agent(source: str, baseline: MLPPolicy) -> RepairAgent:
    if source == "deterministic":
        return DeterministicRepairAgent()
    if source == "llm":
        return OllamaRepairAgent()
    if source.startswith("negative:"):
        return NegativeControlAgent(source.split(":", 1)[1], baseline)
    raise ValueError(f"unknown proposal source {source!r}; choose from {SOURCES}")


def _variant(policy: MLPPolicy, scenario: Scenario, budget: int, guard: bool) -> dict:
    run = TaskAgent(policy, guard=RepeatGuard() if guard else None, budget=budget).run(scenario)
    outcome = evaluate_run(scenario, run).to_dict()
    outcome["trace"] = run.trace
    return outcome


def run_experiment(
    source: str = "deterministic",
    repair_agent: RepairAgent | None = None,
    budget: int = DEFAULT_ACTION_BUDGET,
    constraints: PatchConstraints = PatchConstraints(),
) -> dict:
    baseline = load_baseline_policy()
    fingerprint_before = baseline.fingerprint()
    diagnosis_case = load_diagnosis_case()
    verification_cases = load_verification_cases()  # never passed to the repair agent
    agent = repair_agent or make_repair_agent(source, baseline)

    baseline_run = _variant(baseline, diagnosis_case, budget, guard=False)
    loop_len, loop_start = longest_repeat(baseline_run["trace"])
    report: dict = {
        "app": "Agent Repair Lab",
        "version": __version__,
        "display_version": DISPLAY_VERSION,
        "credit": CREDIT,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "source_label": agent.label,
        "budget": budget,
        "constraints": constraints.to_dict(),
        "diagnosis_case": asdict(diagnosis_case),
        "verification_case_ids": [c.id for c in verification_cases],
        "baseline_loop": {"length": loop_len, "start_step": loop_start},
        "comparison": {
            "baseline": baseline_run,
            "guarded": _variant(baseline, diagnosis_case, budget, guard=True),
            "edited": None,
        },
        "proposal": None,
        "repair_error": None,
        "llm_raw_output": None,
        "validation": None,
        "weight_changes": [],
        "verification": None,
    }

    request = build_repair_request(baseline_run["trace"], baseline, constraints)
    try:
        proposal = agent.propose(request)
    except RepairUnavailable as exc:
        report["repair_error"] = str(exc)
        report["llm_raw_output"] = getattr(agent, "last_raw_output", None)
        return _finish(report, REPAIR_FAILED, baseline, fingerprint_before)
    report["proposal"] = proposal
    report["llm_raw_output"] = getattr(agent, "last_raw_output", None)

    validation = validate_proposal(proposal, baseline, constraints)
    report["validation"] = {"valid": validation.valid, "errors": validation.errors}
    if not validation.valid:
        return _finish(report, INVALID, baseline, fingerprint_before)

    edited = apply_edits(baseline, validation.edits)
    report["weight_changes"] = weight_diff(baseline, edited)
    report["comparison"]["edited"] = _variant(edited, diagnosis_case, budget, guard=False)

    verdict = Verifier(budget).verify(baseline, edited, diagnosis_case, verification_cases)
    report["verification"] = verdict.to_dict()
    report["all_cases"] = _all_cases(baseline, edited, [diagnosis_case, *verification_cases], budget)
    return _finish(report, ACCEPTED if verdict.accepted else REJECTED, baseline, fingerprint_before)


def _all_cases(baseline: MLPPolicy, edited: MLPPolicy, scenarios: list[Scenario], budget: int) -> list[dict]:
    rows = []
    for s in scenarios:
        row = {"scenario_id": s.id, "role": s.role}
        for name, policy, guard in (("baseline", baseline, False), ("guarded", baseline, True), ("edited", edited, False)):
            v = _variant(policy, s, budget, guard)
            row[name] = {k: v[k] for k in ("passed", "termination", "steps", "loop_detected", "failure")}
        rows.append(row)
    return rows


def _finish(report: dict, status: str, baseline: MLPPolicy, fingerprint_before: str) -> dict:
    report["status"] = status
    report["exit_code"] = EXIT_CODES[status]
    report["policy_fingerprint"] = fingerprint_before
    report["original_weights_unchanged"] = baseline.fingerprint() == fingerprint_before
    return report
