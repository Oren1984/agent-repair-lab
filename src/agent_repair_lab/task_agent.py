"""The tool-using task agent. Its only decision maker is the MLP policy.

Baseline, guarded and edited variants all run through ``TaskAgent.run``; the
policy parameters (and the optional repeat guard) are the only differences.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .environment import JobEnvironment, Scenario
from .policy import ACTIONS, FEATURES, MLPPolicy

DEFAULT_ACTION_BUDGET = 10
REPEAT_WINDOW = 3  # identical consecutive (action, observation) steps that count as a loop

RETURNED = "returned_result"
BUDGET_EXHAUSTED = "budget_exhausted"
GUARD_STOPPED = "repeat_guard_stopped"


@dataclass
class AgentMemory:
    is_batch: bool
    last_status: str | None = None
    last_fetch_not_ready: bool = False
    result: str | None = None

    def features(self) -> np.ndarray:
        values = {
            "no_status_yet": self.last_status is None,
            "status_running": self.last_status == "running",
            "status_ready": self.last_status == "ready",
            "is_batch": self.is_batch,
            "has_result": self.result is not None,
            "last_fetch_not_ready": self.last_fetch_not_ready,
        }
        return np.array([float(values[name]) for name in FEATURES])

    def observe(self, tool: str, obs: dict) -> None:
        if tool == "check_status":
            self.last_status = obs.get("status")
            self.last_fetch_not_ready = False
        elif tool == "fetch_result":
            if obs.get("status") == "ok":
                self.result = obs["result"]
                self.last_fetch_not_ready = False
            elif obs.get("status") == "not_ready":
                self.last_fetch_not_ready = True
                self.last_status = "running"


@dataclass
class RunResult:
    scenario_id: str
    answer: str | None
    termination: str
    steps: int
    premature_fetches: int
    trace: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "answer": self.answer,
            "termination": self.termination,
            "steps": self.steps,
            "premature_fetches": self.premature_fetches,
            "trace": self.trace,
        }


class RepeatGuard:
    """Stops a run after REPEAT_WINDOW identical steps in an unchanged state.

    The signature includes the observation, so polling a job whose progress is
    advancing is never treated as a repeat.
    """

    def __init__(self, window: int = REPEAT_WINDOW):
        self.window = window

    def should_stop(self, trace: list[dict]) -> bool:
        if len(trace) < self.window:
            return False
        recent = [step["signature"] for step in trace[-self.window :]]
        return all(sig == recent[0] for sig in recent)


def summarize(tool: str, obs: dict) -> str:
    if tool == "return_result":
        return f"returned {obs['answer']!r}"
    return ", ".join(f"{k}={v}" for k, v in obs.items())


def longest_repeat(trace: list[dict]) -> tuple[int, int]:
    """Length and starting step of the longest run of identical step signatures."""
    best_len, best_start, run_len, run_start = 0, 0, 0, 0
    previous = None
    for step in trace:
        if step["signature"] == previous:
            run_len += 1
        else:
            run_len, run_start = 1, step["step"]
        previous = step["signature"]
        if run_len > best_len:
            best_len, best_start = run_len, run_start
    return best_len, best_start


class TaskAgent:
    def __init__(self, policy: MLPPolicy, guard: RepeatGuard | None = None, budget: int = DEFAULT_ACTION_BUDGET):
        self.policy = policy
        self.guard = guard
        self.budget = budget

    def run(self, scenario: Scenario) -> RunResult:
        env = JobEnvironment(scenario)
        memory = AgentMemory(is_batch=scenario.queue == "batch")
        args = {"job_id": scenario.job_id}
        trace: list[dict] = []
        termination, answer = BUDGET_EXHAUSTED, None

        for step in range(1, self.budget + 1):
            x = memory.features()
            out = self.policy.forward(x)
            tool = ACTIONS[out.action_index]

            if tool == "check_status":
                obs = env.check_status(args["job_id"])
            elif tool == "fetch_result":
                obs = env.fetch_result(args["job_id"])
            else:
                answer = memory.result
                obs = {"answer": answer}

            trace.append(
                {
                    "step": step,
                    "tool": tool,
                    "args": dict(args) if tool != "return_result" else {},
                    "result_summary": summarize(tool, obs),
                    "features": {name: float(v) for name, v in zip(FEATURES, x)},
                    "hidden": [round(float(v), 6) for v in out.hidden],
                    "scores": {name: round(float(v), 6) for name, v in zip(ACTIONS, out.scores)},
                    "signature": json.dumps([tool, args, obs, memory.result is not None], sort_keys=True),
                }
            )
            if tool == "return_result":
                termination = RETURNED
                break
            memory.observe(tool, obs)
            if self.guard is not None and self.guard.should_stop(trace):
                termination = GUARD_STOPPED
                break

        return RunResult(
            scenario_id=scenario.id,
            answer=answer,
            termination=termination,
            steps=len(trace),
            premature_fetches=env.premature_fetches,
            trace=trace,
        )
