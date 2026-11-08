"""Optional LLM mode, tested offline with simulated HTTP responses."""

import json
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agent_repair_lab.environment import load_diagnosis_case
from agent_repair_lab.experiment import run_experiment
from agent_repair_lab.llm_adapter import OllamaConfig, OllamaRepairAgent, parse_model_output
from agent_repair_lab.patch import PatchConstraints
from agent_repair_lab.policy import load_baseline_policy
from agent_repair_lab.repair_agent import DeterministicRepairAgent, RepairUnavailable, build_repair_request
from agent_repair_lab.task_agent import TaskAgent

CONFIG = OllamaConfig(url="http://llm.invalid", model="test-model", timeout=1)


def _request():
    base = load_baseline_policy()
    return build_repair_request(TaskAgent(base).run(load_diagnosis_case()).trace, base, PatchConstraints())


def _reply(content: str):
    return lambda url, payload, timeout: {"message": {"role": "assistant", "content": content}}


def _good_content():
    return json.dumps(DeterministicRepairAgent().propose(_request()))


def test_payload_contains_only_repair_inputs():
    captured = {}

    def transport(url, payload, timeout):
        captured.update(url=url, payload=payload, timeout=timeout)
        return {"message": {"content": _good_content()}}

    OllamaRepairAgent(CONFIG, transport).propose(_request())
    assert captured["url"] == "http://llm.invalid/api/chat"
    assert captured["timeout"] == 1
    body = captured["payload"]["messages"][1]["content"]
    for forbidden in ("J-2077", "J-3300", "J-3815", "J-4105", "expected_result", "verification"):
        assert forbidden not in body
    assert "failing_trace" in body and "allowed_slice" in body and "constraints" in body


def test_valid_llm_output_goes_through_same_validation_and_verifier():
    agent = OllamaRepairAgent(CONFIG, _reply("```json\n" + _good_content() + "\n```"))
    report = run_experiment("llm", repair_agent=agent)
    assert report["source_label"] == "AI agent repair (local LLM)"
    assert report["validation"]["valid"]
    assert report["status"] == "accepted"
    assert report["llm_raw_output"].startswith("```json")


def test_llm_proposal_that_fails_verification_is_rejected():
    base = load_baseline_policy()
    bad = {
        "diagnosis": "return early",
        "edits": [{"parameter": "W2", "index": [2, 3], "expected_original": float(base.params["W2"][2, 3]), "new_value": 2.0}],
    }
    report = run_experiment("llm", repair_agent=OllamaRepairAgent(CONFIG, _reply(json.dumps(bad))))
    assert report["status"] == "rejected" and report["exit_code"] == 1


def test_llm_output_with_wrong_original_is_invalid():
    bad = {"diagnosis": "x", "edits": [{"parameter": "W2", "index": [1, 3], "expected_original": 0.9, "new_value": 1.5}]}
    report = run_experiment("llm", repair_agent=OllamaRepairAgent(CONFIG, _reply(json.dumps(bad))))
    assert report["status"] == "invalid_proposal"
    assert report["comparison"]["edited"] is None


@pytest.mark.parametrize("content", ["I think you should raise the weight.", "{not json", "[1, 2]"])
def test_invalid_llm_output_is_reported_not_substituted(content):
    report = run_experiment("llm", repair_agent=OllamaRepairAgent(CONFIG, _reply(content)))
    assert report["status"] == "repair_failed"
    assert report["proposal"] is None and report["weight_changes"] == []
    assert report["llm_raw_output"] == content
    assert report["repair_error"]


@pytest.mark.parametrize(
    "exc",
    [urllib.error.URLError("connection refused"), TimeoutError("timed out"), urllib.error.HTTPError("u", 404, "nf", {}, None)],
)
def test_service_failures_are_reported(exc):
    def transport(url, payload, timeout):
        raise exc

    with pytest.raises(RepairUnavailable):
        OllamaRepairAgent(CONFIG, transport).propose(_request())
    report = run_experiment("llm", repair_agent=OllamaRepairAgent(CONFIG, transport))
    assert report["status"] == "repair_failed" and report["exit_code"] == 2


def test_missing_message_content_is_reported():
    with pytest.raises(RepairUnavailable):
        OllamaRepairAgent(CONFIG, lambda u, p, t: {"error": "model not found"}).propose(_request())


def test_parse_model_output_handles_fenced_json():
    assert parse_model_output('```json\n{"a": 1}\n```') == {"a": 1}


def test_real_http_path_against_local_stub_server():
    """Exercises the urllib transport end to end against an in-process HTTP stub."""
    content = _good_content()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers["Content-Length"])
            json.loads(self.rfile.read(length))
            body = json.dumps({"message": {"role": "assistant", "content": content}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = OllamaConfig(url=f"http://127.0.0.1:{server.server_port}", model="stub", timeout=5)
        report = run_experiment("llm", repair_agent=OllamaRepairAgent(config))
        assert report["status"] == "accepted"
    finally:
        server.shutdown()


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("ARL_OLLAMA_URL", "http://host.docker.internal:11434/")
    monkeypatch.setenv("ARL_OLLAMA_MODEL", "qwen2.5")
    monkeypatch.setenv("ARL_OLLAMA_TIMEOUT", "30")
    cfg = OllamaConfig.from_env()
    assert cfg.url == "http://host.docker.internal:11434" and cfg.model == "qwen2.5" and cfg.timeout == 30
