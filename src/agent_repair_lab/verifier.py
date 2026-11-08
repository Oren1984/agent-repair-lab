"""Independent verifier. It owns the verdict.

It imports nothing from the repair agents. The experiment runner hands it the
candidate policy, the baseline policy and the verification fixtures directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .environment import Scenario
from .policy import MLPPolicy
from .task_agent import REPEAT_WINDOW, RETURNED, RunResult, TaskAgent, longest_repeat


@dataclass
class CaseOutcome:
    scenario_id: str
    role: str
    expected: str
    answer: str | None
    correct: bool
    loop_detected: bool
    steps: int
    termination: str
    premature_fetches: int
    passed: bool
    failure: str | None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def evaluate_run(scenario: Scenario, run: RunResult) -> CaseOutcome:
    repeat_len, _ = longest_repeat(run.trace)
    loop = repeat_len >= REPEAT_WINDOW
    correct = run.termination == RETURNED and run.answer == scenario.expected_result
    failure = None
    if run.premature_fetches:
        failure = f"premature fetch_result ({run.premature_fetches}x before the job was ready)"
    elif run.termination != RETURNED:
        failure = f"did not complete ({run.termination})"
    elif not correct:
        failure = f"wrong result: returned {run.answer!r}"
    elif loop:
        failure = f"loop detected ({repeat_len} identical steps)"
    return CaseOutcome(
        scenario_id=scenario.id,
        role=scenario.role,
        expected=scenario.expected_result,
        answer=run.answer,
        correct=correct,
        loop_detected=loop,
        steps=run.steps,
        termination=run.termination,
        premature_fetches=run.premature_fetches,
        passed=failure is None,
        failure=failure,
    )


@dataclass
class Verdict:
    accepted: bool
    reasons: list[str] = field(default_factory=list)
    candidate: list[CaseOutcome] = field(default_factory=list)
    baseline: list[CaseOutcome] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "reasons": self.reasons,
            "candidate": [c.to_dict() for c in self.candidate],
            "baseline": [c.to_dict() for c in self.baseline],
        }


class Verifier:
    """Acceptance rules:

    * the diagnosis case and every held-out failure case complete with the
      expected result and no loop;
    * every delayed-ready case completes with no premature fetch;
    * every regression case (which the baseline completes) still completes.
    Any failure, premature fetch, timeout or regression is a rejection.
    """

    def __init__(self, budget: int):
        self.budget = budget

    def verify(self, baseline: MLPPolicy, candidate: MLPPolicy, diagnosis_case: Scenario, cases: list[Scenario]) -> Verdict:
        scenarios = [diagnosis_case, *cases]
        base = [evaluate_run(s, TaskAgent(baseline, budget=self.budget).run(s)) for s in scenarios]
        cand = [evaluate_run(s, TaskAgent(candidate, budget=self.budget).run(s)) for s in scenarios]

        reasons: list[str] = []
        for b, c in zip(base, cand):
            if c.passed:
                continue
            if c.role == "regression" and b.passed:
                reasons.append(f"REGRESSION on {c.scenario_id}: baseline passed, candidate {c.failure}")
            else:
                reasons.append(f"{c.role} case {c.scenario_id}: {c.failure}")
        if cand[0].loop_detected:
            reasons.append("the demonstrated loop is still present")

        accepted = not reasons
        if accepted:
            reasons.append(
                f"all {len(cand)} cases passed: diagnosis + held-out failure fixed, "
                "no premature fetch on delayed-ready, no regression"
            )
        return Verdict(accepted=accepted, reasons=reasons, candidate=cand, baseline=base)
