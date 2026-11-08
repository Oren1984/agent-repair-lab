"""Agent Repair Lab — single Streamlit screen. All logic lives in agent_repair_lab.experiment."""

from __future__ import annotations

import json

import streamlit as st

from agent_repair_lab import CREDIT, DISPLAY_VERSION
from agent_repair_lab.experiment import run_experiment
from agent_repair_lab.llm_adapter import OllamaConfig

SOURCE_OPTIONS = {
    "Deterministic repair (default, offline)": "deterministic",
    "AI agent repair — local LLM via Ollama (optional)": "llm",
    "Negative control: force return (bad patch)": "negative:force-return",
    "Negative control: force fetch (bad patch)": "negative:force-fetch",
    "Negative control: rebalance (bad patch)": "negative:rebalance",
    "Negative control: excessive edit (bad patch)": "negative:excessive",
}
STATUS_TEXT = {
    "accepted": "ACCEPTED — the independent verifier accepted the repair.",
    "rejected": "REJECTED — the patch was applied to a copy but failed independent verification.",
    "invalid_proposal": "INVALID PROPOSAL — patch validation failed; no weights were written.",
    "repair_failed": "REPAIR AGENT FAILED — no proposal was produced; nothing was substituted.",
}
VARIANTS = (
    ("baseline", "1 · Baseline policy", "Original weights, no guard."),
    ("guarded", "2 · Baseline + repeat guard", "Same weights; a guard stops repeated identical steps."),
    ("edited", "3 · Edited policy", "Repaired weights, no guard, same budget."),
)

st.set_page_config(page_title="Agent Repair Lab", layout="wide")
st.title(f"Agent Repair Lab {DISPLAY_VERSION}")
st.markdown(
    "**One agent gets stuck.  \nAnother agent proposes a weight repair.  \n"
    "An independent verifier decides whether it worked.**"
)

tab_experiment, tab_help = st.tabs(["Experiment", "Help Center"])

with tab_experiment:
    with st.container(border=True):
        left, right = st.columns([3, 1])
        with left:
            label = st.selectbox("Repair agent", list(SOURCE_OPTIONS), index=0)
            source = SOURCE_OPTIONS[label]
            if source == "llm":
                cfg = OllamaConfig.from_env()
                st.info(
                    f"Optional local LLM mode. Endpoint `{cfg.url}`, model `{cfg.model}`, timeout {cfg.timeout:.0f}s "
                    "(set ARL_OLLAMA_URL / ARL_OLLAMA_MODEL). If Ollama is unavailable the failure is shown — "
                    "no deterministic result is substituted."
                )
            elif source.startswith("negative:"):
                st.warning("Negative control: a hand-crafted bad patch used to show the verifier rejecting it.")
        with right:
            st.write("")
            st.write("")
            run_clicked = st.button("Run experiment", type="primary", use_container_width=True)

    if run_clicked:
        st.session_state.pop("report", None)  # never show an older run next to a new attempt
        st.session_state["run_count"] = st.session_state.get("run_count", 0) + 1
        with st.spinner("Running baseline, repair agent and verifier…"):
            try:
                st.session_state["report"] = run_experiment(source)
            except Exception as exc:  # unexpected internal error: show it, keep nothing stale
                st.session_state["report"] = {"internal_error": f"{type(exc).__name__}: {exc}"}

report = st.session_state.get("report")


def _bool(v: bool) -> str:
    return "yes" if v else "no"


def render(report: dict) -> None:
    run_no = st.session_state.get("run_count", 1)
    if "internal_error" in report:
        st.error(f"Run #{run_no} crashed: {report['internal_error']}")
        return
    st.caption(f"Run #{run_no} · {report['timestamp']} · repair agent: {report['source_label']}")
    status = report["status"]
    (st.success if status == "accepted" else st.error)(STATUS_TEXT[status])

    case = report["diagnosis_case"]
    st.subheader("Three-way comparison on the diagnosis case")
    st.caption(
        f"Case `{case['id']}` — job {case['job_id']} on the {case['queue']} queue, "
        f"expected result: “{case['expected_result']}”. Action budget: {report['budget']}."
    )
    cols = st.columns(3)
    for col, (key, title, sub) in zip(cols, VARIANTS):
        v = report["comparison"][key]
        with col, st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(sub)
            if v is None:
                st.write("Not run — no valid patch.")
                continue
            verdict = "Completed correctly" if v["passed"] else ("Stopped" if v["termination"] == "repeat_guard_stopped" else "Failed")
            st.metric("Outcome", verdict)
            st.text(
                f"termination : {v['termination']}\n"
                f"actions     : {v['steps']}\n"
                f"loop        : {_bool(v['loop_detected'])}\n"
                f"premature   : {v['premature_fetches']}\n"
                f"result      : {v['answer']!r}"
            )

    loop = report["baseline_loop"]
    with st.expander(
        f"Baseline trace — repeats the same step {loop['length']}× from step {loop['start_step']}", expanded=True
    ):
        rows = []
        prev = None
        for s in report["comparison"]["baseline"]["trace"]:
            repeat = s["signature"] == prev
            prev = s["signature"]
            rows.append(
                {
                    "step": s["step"],
                    "tool": s["tool"],
                    "args": json.dumps(s["args"]),
                    "result": s["result_summary"],
                    "check_status": round(s["scores"]["check_status"], 3),
                    "fetch_result": round(s["scores"]["fetch_result"], 3),
                    "return_result": round(s["scores"]["return_result"], 3),
                    "repeat": "⟲ same as previous" if repeat else "",
                }
            )
        st.dataframe(rows, hide_index=True, use_container_width=True)
        st.caption("Scores are the policy's output-layer action scores; the highest score is executed.")

    st.subheader("Repair agent proposal")
    if report["repair_error"]:
        st.error(f"Repair agent failed: {report['repair_error']}")
    if report["proposal"] is not None:
        st.markdown("**Diagnosis** (rendered as plain text)")
        st.text(str(report["proposal"].get("diagnosis", "")))
    if report["llm_raw_output"] is not None:
        with st.expander("Raw LLM output (plain text)"):
            st.text(report["llm_raw_output"])
    validation = report["validation"]
    if validation and not validation["valid"]:
        st.error("Patch validation failed before any weight was written:")
        st.text("\n".join(f"- {e}" for e in validation["errors"]))
        with st.expander("Rejected proposal payload"):
            st.text(json.dumps(report["proposal"], indent=2))
    if report["weight_changes"]:
        st.markdown("**Weight edits applied to an in-memory copy**")
        st.dataframe(
            [
                {
                    "parameter": f"{c['parameter']}{c['index']}",
                    "meaning": c["label"],
                    "before": round(c["before"], 3),
                    "after": round(c["after"], 3),
                    "delta": f"{c['delta']:+.3f}",
                }
                for c in report["weight_changes"]
            ],
            hide_index=True,
            use_container_width=True,
        )
    st.caption(
        f"Original weights unchanged after this run: {_bool(report['original_weights_unchanged'])} "
        f"(fingerprint {report['policy_fingerprint']}). Allowlist: W2 only · "
        f"max {report['constraints']['max_edits']} edits · max |delta| {report['constraints']['max_abs_delta']}."
    )

    st.subheader("Independent verification")
    ver = report["verification"]
    if ver is None:
        st.write("Not run — there is no valid patch to verify.")
        return
    base = {c["scenario_id"]: c for c in ver["baseline"]}
    st.dataframe(
        [
            {
                "case": c["scenario_id"],
                "role": c["role"],
                "baseline": "pass" if base[c["scenario_id"]]["passed"] else "fail",
                "edited": "PASS" if c["passed"] else "FAIL",
                "correct result": _bool(c["correct"]),
                "loop": _bool(c["loop_detected"]),
                "actions": c["steps"],
                "termination": c["termination"],
                "premature fetch": c["premature_fetches"],
                "failure": c["failure"] or "",
            }
            for c in ver["candidate"]
        ],
        hide_index=True,
        use_container_width=True,
    )
    st.markdown("**Verdict: " + ("ACCEPTED" if ver["accepted"] else "REJECTED") + "**")
    st.text("\n".join(f"- {r}" for r in ver["reasons"]))
    with st.expander("All cases × all three variants"):
        st.dataframe(
            [
                {
                    "case": r["scenario_id"],
                    "role": r["role"],
                    **{f"{k}": ("pass" if r[k]["passed"] else f"fail ({r[k]['termination']})") for k in ("baseline", "guarded", "edited")},
                }
                for r in report["all_cases"]
            ],
            hide_index=True,
            use_container_width=True,
        )


REPO_URL = "https://github.com/Oren1984/agent-repair-lab"
RESEARCH_URL = "https://rikiyat.github.io/blog/weight-space-programming/"
VIDEO_URL = "https://www.youtube.com/watch?v=VAbXapGNveU"


def render_help() -> None:
    """Static help text only: no experiment runs and no network calls happen here."""
    st.subheader("Help Center")
    st.caption("A short guide for using this demo live. Switching tabs keeps your latest result.")

    with st.expander("1 · Start here", expanded=True):
        st.markdown(
            "1. **Run the demo.** On the **Experiment** tab, keep *Deterministic repair* and press **Run experiment**.\n"
            "2. **Compare the three outcomes.** The three cards show the same task run by the original policy, "
            "the original policy with a repeat guard, and the edited policy.\n"
            "3. **Open the details.** Scroll down for the baseline trace, the proposed weight edit, "
            "and the independent verification table with its verdict."
        )

    with st.expander("2 · Read the results"):
        st.markdown(
            "- **Baseline policy — Failed.** The agent keeps calling `check_status` after the job reports ready, "
            "until its 10-action budget runs out. The trace marks each repeated step.\n"
            "- **Baseline + repeat guard — Stopped.** A guard ends the run after 3 identical steps. "
            "That saves wasted work, but the agent never fetched the result, so the task is **still incomplete**.\n"
            "- **Edited policy — Completed correctly.** One output-layer weight was changed. "
            "The agent fetches and returns the expected result, with no guard and the same budget.\n\n"
            "Stopping a loop is not the same as finishing the task."
        )

    with st.expander("3 · Why a repair can be rejected"):
        st.markdown(
            "A repair must fix the failure **and** keep behaviour that already worked. "
            "Choose **Negative control: rebalance** and run it:\n\n"
            "- it fixes the demonstrated batch-job loop and the held-out batch case;\n"
            "- but it takes weight away from interactive jobs, so both interactive regression cases now loop "
            "until the budget runs out;\n"
            "- the verifier reports **REGRESSION** and rejects the patch.\n\n"
            "Other negative controls are rejected for returning the wrong result (*force return*), "
            "fetching before the job is ready (*force fetch*), or exceeding the patch limits before any weight "
            "is written (*excessive edit*)."
        )

    with st.expander("4 · Repair modes"):
        st.markdown(
            "- **Deterministic repair (default, offline).** A rule-based repair agent reads the failing trace and "
            "the editable output-layer weights and computes one bounded numeric edit. No LLM is involved.\n"
            "- **AI agent repair — local LLM (optional).** Sends the same inputs to a local Ollama model "
            "(`ARL_OLLAMA_URL`, `ARL_OLLAMA_MODEL`). Its proposal goes through the same validation and verification.\n"
            "- **If Ollama is unavailable,** the run shows *REPAIR AGENT FAILED — no proposal was produced; nothing "
            "was substituted*. No deterministic result is shown under the LLM label.\n\n"
            "Status: the LLM path is tested offline with simulated responses. A run against a live local model "
            "has **not** been verified yet."
        )

    with st.expander("5 · Sources and documentation"):
        st.markdown(
            f"- Primary research — [*Sparks of Weight-Space Programming: LLMs can natively read & write neural "
            f"network weights*]({RESEARCH_URL}) (Takehi, Teoh, Zhang, Khattab; preliminary research).\n"
            f"- Video inspiration — [Code Cave, *ניתוח מוח פתוח: מה קורה כש-AI מתקן את המוח של AI אחר?*]({VIDEO_URL})\n"
            f"- Project documentation — [{REPO_URL}/tree/main/docs]({REPO_URL}/tree/main/docs)\n\n"
            "Agent Repair Lab is a separate small-policy experiment inspired by that research. "
            "It does not reproduce the researchers' results; its own measurements are shown on the Experiment tab."
        )


with tab_experiment:
    if report:
        render(report)
    else:
        st.info("Press **Run experiment** to reproduce the loop, propose a repair and verify it.")

with tab_help:
    render_help()

st.divider()
st.caption(CREDIT)
