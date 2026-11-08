"""Baseline failure, normal behaviour and the repeat guard."""

import numpy as np
import pytest

from agent_repair_lab.environment import load_diagnosis_case, load_verification_cases
from agent_repair_lab.policy import ACTIONS, MLPPolicy, load_baseline_policy
from agent_repair_lab.task_agent import (
    BUDGET_EXHAUSTED,
    DEFAULT_ACTION_BUDGET,
    GUARD_STOPPED,
    RETURNED,
    RepeatGuard,
    TaskAgent,
    longest_repeat,
)
from agent_repair_lab.verifier import evaluate_run

CASES = {c.id: c for c in load_verification_cases()}


def test_baseline_reproduces_loop_on_diagnosis_case():
    case = load_diagnosis_case()
    run = TaskAgent(load_baseline_policy()).run(case)
    assert run.termination == BUDGET_EXHAUSTED
    assert run.steps == DEFAULT_ACTION_BUDGET
    assert run.answer is None
    # after the job reports ready, the agent keeps calling check_status
    ready_steps = [s for s in run.trace if "status=ready" in s["result_summary"]]
    assert len(ready_steps) == DEFAULT_ACTION_BUDGET - 1
    assert all(s["tool"] == "check_status" for s in run.trace)
    assert longest_repeat(run.trace)[0] >= 3


def test_baseline_reproduction_is_deterministic():
    case = load_diagnosis_case()
    a = TaskAgent(load_baseline_policy()).run(case)
    b = TaskAgent(load_baseline_policy()).run(case)
    assert a.trace == b.trace


@pytest.mark.parametrize("case_id", ["regression-interactive-J3300", "regression-interactive-slow-J3815"])
def test_baseline_completes_normal_scenarios(case_id):
    case = CASES[case_id]
    run = TaskAgent(load_baseline_policy()).run(case)
    outcome = evaluate_run(case, run)
    assert outcome.passed, outcome.failure
    assert run.answer == case.expected_result


def test_baseline_waits_while_job_is_running():
    case = CASES["regression-interactive-slow-J3815"]
    run = TaskAgent(load_baseline_policy()).run(case)
    tools = [s["tool"] for s in run.trace]
    assert tools == ["check_status"] * 5 + ["fetch_result", "return_result"]
    assert run.premature_fetches == 0


def test_repeat_guard_stops_loop_but_task_is_not_completed():
    case = load_diagnosis_case()
    run = TaskAgent(load_baseline_policy(), guard=RepeatGuard()).run(case)
    assert run.termination == GUARD_STOPPED
    assert run.answer is None
    assert not evaluate_run(case, run).passed
    assert run.steps < DEFAULT_ACTION_BUDGET


def test_repeat_guard_does_not_stop_legitimate_polling():
    case = CASES["regression-interactive-slow-J3815"]
    run = TaskAgent(load_baseline_policy(), guard=RepeatGuard()).run(case)
    assert run.termination == RETURNED
    assert run.answer == case.expected_result


def test_action_genuinely_depends_on_weights():
    base = load_baseline_policy()
    x = np.array([0, 0, 1, 1, 0, 0], dtype=float)  # ready, batch, no result
    assert base.forward(x).action == "check_status"
    params = {k: v.copy() for k, v in base.params.items()}
    params["W2"][ACTIONS.index("fetch_result"), 3] = 2.0
    assert MLPPolicy(params).forward(x).action == "fetch_result"
