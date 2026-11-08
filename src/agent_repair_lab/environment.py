"""Deterministic simulated job service the task agent works against."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources


@dataclass(frozen=True)
class Scenario:
    id: str
    role: str  # diagnosis | held_out_failure | regression | delayed_ready
    description: str
    job_id: str
    queue: str  # "batch" or "interactive"
    ready_at_tick: int  # first tool-call tick at which the job reports ready
    result_payload: str  # what the simulated service stores for the job
    expected_result: str  # declared independently by the fixture author

    @classmethod
    def from_dict(cls, raw: dict) -> "Scenario":
        return cls(**{field: raw[field] for field in cls.__dataclass_fields__})


class JobEnvironment:
    """A job that becomes ready after a fixed number of tool calls.

    Every tool call advances the clock by one tick, so legitimate polling sees
    the progress value change while the job is running.
    """

    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.tick = 0
        self.premature_fetches = 0

    @property
    def ready(self) -> bool:
        return self.tick >= self.scenario.ready_at_tick

    def _progress(self) -> int:
        if self.ready:
            return 100
        return int(100 * self.tick / self.scenario.ready_at_tick)

    def check_status(self, job_id: str) -> dict:
        if job_id != self.scenario.job_id:
            obs = {"status": "unknown_job"}
        elif self.ready:
            obs = {"status": "ready", "progress": 100}
        else:
            obs = {"status": "running", "progress": self._progress()}
        self.tick += 1
        return obs

    def fetch_result(self, job_id: str) -> dict:
        if job_id != self.scenario.job_id:
            obs = {"status": "unknown_job"}
        elif self.ready:
            obs = {"status": "ok", "result": self.scenario.result_payload}
        else:
            self.premature_fetches += 1
            obs = {"status": "not_ready", "progress": self._progress()}
        self.tick += 1
        return obs


def _load(name: str) -> list[Scenario]:
    text = resources.files("agent_repair_lab.fixtures").joinpath(name).read_text()
    return [Scenario.from_dict(item) for item in json.loads(text)["scenarios"]]


def load_diagnosis_case() -> Scenario:
    """The single failing case the repair agent is allowed to see."""
    return _load("diagnosis_case.json")[0]


def load_verification_cases() -> list[Scenario]:
    """Held-out cases; supplied only to the verifier, never to a repair agent."""
    return _load("verification_cases.json")
