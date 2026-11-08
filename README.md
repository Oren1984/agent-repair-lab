# Agent Repair Lab

**V1.0 · AI Research & Applied POCs**

> One agent gets stuck.
> Another agent proposes a weight repair.
> An independent verifier decides whether it worked.

Agent Repair Lab is a small, local research proof of concept. It asks one question: **can one AI agent diagnose why another agent is stuck, propose a targeted numeric change to the neural policy driving it, and produce a repair that passes independent verification?**

A repair agent can propose any change it likes. This project's contribution is the **verification workflow around the repair**: the edit is bounded and validated, it is applied only to a copy, and an independent verifier decides the outcome using held-out cases, regression cases and a delayed-ready case. A patch that ends the loop by taking the wrong action is rejected.

![Baseline loop vs. repeat guard vs. edited policy](docs/images/01-baseline-loop.png)

## What it demonstrates

- **Task agent.** A tool-using agent checks the status of a simulated job, fetches the result once the job is ready, and returns it. A real 6→7→3 NumPy MLP picks every action. Nothing else in the code chooses actions.
- **Reproducible failure.** One output-layer weight in the baseline is faulty. On *batch* jobs the agent keeps calling `check_status` after the job reports ready, until it runs out of actions.
- **Three variants, one runner.** All three run on the same environment, tools and execution path:
  1. **Baseline**: loops until the 10-action budget is exhausted.
  2. **Baseline + repeat guard**: stops after 3 identical steps in an unchanged state. The loop stops, but the task is **not** completed.
  3. **Edited policy**: the repaired weights, with no guard and the same budget. It completes the task with the correct result.
- **Repair agent.** It reads the failing trace and the editable `W2` slice and proposes a diagnosis plus bounded numeric edits.
  - **Deterministic repair** (default, offline): a rule-based agent that computes the edit from the observed action scores and hidden activations.
  - **AI agent repair: local LLM** (optional): an Ollama adapter that sends the same inputs to a local model. Its proposal goes through exactly the same validation and verification.
- **Independent verifier.** It runs fixtures the repair agent never sees, and it alone gives the verdict: *accepted*, *rejected*, *invalid proposal* or *repair agent failed*.
- **Negative controls.** Four hand-crafted bad patches show the verifier at work. *force-return* returns the wrong result, *force-fetch* fetches prematurely, *rebalance* causes a regression, and *excessive* is blocked before any weight is written.

Observed result of the default run: one weight changed, `W2[fetch_result ← ready_batch]` from `0.187` to `1.368`. All 5 verification cases passed and the verifier **accepted** the repair. Details are in [docs/EXPERIMENT.md](docs/EXPERIMENT.md).

## Run it

### Docker (UI)

```bash
docker compose up --build
```

Open http://localhost:8501 and press **Run experiment**. To run alongside other Streamlit apps, choose a different host port:

```bash
AGENT_REPAIR_LAB_PORT=8502 docker compose up --build
```

On Windows PowerShell, use `$env:AGENT_REPAIR_LAB_PORT=8502; docker compose up --build`.

### Local (Python 3.11+)

```bash
python -m venv .venv
.venv/Scripts/activate        # Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                     # offline test suite
streamlit run app.py          # UI
```

### Headless experiment (same runner and verifier as the UI)

```bash
python -m agent_repair_lab                              # readable report, deterministic repair
python -m agent_repair_lab --json --out report.json     # JSON report (add --full for traces)
python -m agent_repair_lab --source negative:rebalance  # negative control
```

Sources: `deterministic`, `llm`, `negative:force-return`, `negative:force-fetch`, `negative:rebalance`, `negative:excessive`.
Exit codes: `0` accepted · `1` rejected by the verifier · `2` invalid proposal or repair agent failure.

### Optional: local LLM repair via Ollama

The default demo and the tests never contact an LLM. To try AI agent repair:

1. Run [Ollama](https://ollama.com) on the host and pull a model, for example `ollama pull llama3.1`.
2. Configure the model through environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `ARL_OLLAMA_URL` | `http://localhost:11434` locally, `http://host.docker.internal:11434` in Compose | Ollama endpoint |
| `ARL_OLLAMA_MODEL` | `llama3.1` | model name |
| `ARL_OLLAMA_TIMEOUT` | `120` | request timeout in seconds |

3. In the UI choose **AI agent repair — local LLM**, or run `python -m agent_repair_lab --source llm`.

**Docker Desktop (Windows/macOS)** resolves `host.docker.internal` to the host. On Linux, Compose maps it through `host-gateway`. If the container cannot reach Ollama, start Ollama with `OLLAMA_HOST=0.0.0.0:11434` so it listens beyond loopback.

If Ollama is unreachable, or returns output that is not valid JSON, the run is reported as **repair agent failed**. No deterministic result is substituted under the LLM label.

## Project layout

```
app.py                         Streamlit screen (thin)
src/agent_repair_lab/
  policy.py                    NumPy MLP, feature/hidden/action definitions
  environment.py               simulated job service + fixture loading
  task_agent.py                agent loop, trace, repeat guard
  repair_agent.py              repair interface, deterministic repair, negative controls
  llm_adapter.py               optional Ollama repair agent
  patch.py                     patch schema, validation, apply-to-copy, weight diff
  verifier.py                  independent verifier and acceptance rules
  experiment.py                experiment runner shared by UI, CLI and tests
  cli.py                       headless runner
  fixtures/                    baseline weights, diagnosis case, verification cases
tests/                         offline pytest suite
docs/EXPERIMENT.md             experiment design, observed results and limits
```

## Sources and Inspiration

1. **Primary research source.** Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab, *"Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights,"* November 2026.
   https://rikiyat.github.io/blog/weight-space-programming/
2. **Video that inspired this project.** Code Cave, *"ניתוח מוח פתוח: מה קורה כש-AI מתקן את המוח של AI אחר?"*
   https://www.youtube.com/watch?v=VAbXapGNveU

That MIT CSAIL work is preliminary research. Any benchmark figures quoted from it in [docs/EXPERIMENT.md](docs/EXPERIMENT.md), including the Qwen figures, are the researchers' own results. Agent Repair Lab V1.0 is an independent toy-policy experiment: it does not reproduce their Qwen benchmark, and its own measured results are reported separately. Its contribution is the **independent verification of proposed repairs**, using held-out cases and regression checks.

---

Built and developed by Oren Salami · V1.0
