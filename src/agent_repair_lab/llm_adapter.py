"""Optional local LLM repair agent (Ollama). Not used by the default demo.

Configuration (environment variables):
  ARL_OLLAMA_URL      default http://localhost:11434
  ARL_OLLAMA_MODEL    default llama3.1
  ARL_OLLAMA_TIMEOUT  seconds, default 120

The model receives only the RepairRequest (failing trace, feature/action
definitions, the editable W2 slice and patch constraints). It gets no verifier
fixtures, no file access and no execution capability; its JSON output is
treated as data and goes through the same validation and verification as any
other proposal. Failures raise ``RepairUnavailable``; nothing is substituted.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable

from .repair_agent import RepairRequest, RepairUnavailable

Transport = Callable[[str, dict, float], dict]

SYSTEM_PROMPT = """You are a repair agent. A task agent driven by a tiny MLP policy is stuck in a loop.
You receive its failing trace (features, hidden activations and action scores per step), definitions
of the features, actions and hidden units, the editable output-layer weights W2 (rows = actions,
columns = hidden units) and the patch constraints.

Diagnose why the agent is stuck and propose a small numeric edit to W2 that makes it take the
appropriate action in the stuck state without changing its behaviour elsewhere (it must still wait
while a job is running and must never return without a result).

Reply with JSON only, exactly in this shape:
{"diagnosis": "<short evidence-based explanation>",
 "edits": [{"parameter": "W2", "index": [<row>, <col>], "expected_original": <current value>, "new_value": <proposed value>}]}
"""


@dataclass(frozen=True)
class OllamaConfig:
    url: str = "http://localhost:11434"
    model: str = "llama3.1"
    timeout: float = 120.0

    @classmethod
    def from_env(cls) -> "OllamaConfig":
        return cls(
            url=os.environ.get("ARL_OLLAMA_URL", cls.url).rstrip("/"),
            model=os.environ.get("ARL_OLLAMA_MODEL", cls.model),
            timeout=float(os.environ.get("ARL_OLLAMA_TIMEOUT", cls.timeout)),
        )


def urllib_transport(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - configured local endpoint
        return json.loads(response.read().decode())


def parse_model_output(text: str) -> dict:
    """Extract the proposal object from the model's reply text."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{") :] if "{" in text else text
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise RepairUnavailable("model output contains no JSON object")
    try:
        proposal = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise RepairUnavailable(f"model output is not valid JSON: {exc.msg}") from exc
    if not isinstance(proposal, dict):
        raise RepairUnavailable("model output JSON is not an object")
    return proposal


class OllamaRepairAgent:
    label = "AI agent repair (local LLM)"

    def __init__(self, config: OllamaConfig | None = None, transport: Transport = urllib_transport):
        self.config = config or OllamaConfig.from_env()
        self.transport = transport
        self.last_raw_output: str | None = None

    def build_payload(self, request: RepairRequest) -> dict:
        return {
            "model": self.config.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(request.to_dict())},
            ],
        }

    def propose(self, request: RepairRequest) -> dict:
        url = f"{self.config.url}/api/chat"
        try:
            reply = self.transport(url, self.build_payload(request), self.config.timeout)
        except urllib.error.HTTPError as exc:
            raise RepairUnavailable(f"Ollama returned HTTP {exc.code} for model {self.config.model!r}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RepairUnavailable(f"Ollama is unreachable at {self.config.url}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise RepairUnavailable("Ollama response is not JSON") from exc
        content = (reply.get("message") or {}).get("content") if isinstance(reply, dict) else None
        if not isinstance(content, str):
            raise RepairUnavailable("Ollama response has no message content")
        self.last_raw_output = content
        return parse_model_output(content)
