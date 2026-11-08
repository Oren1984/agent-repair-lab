"""Focused UI regression test for the Streamlit page (no browser, no network)."""

import urllib.request
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")
CREDIT = "Built and developed by Oren Salami · V1.0"
HELP_SECTIONS = [
    "1 · Start here",
    "2 · Read the results",
    "3 · Why a repair can be rejected",
    "4 · Repair modes",
    "5 · Sources and documentation",
]


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the page must not open network connections")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)


def _texts(at):
    return [m.value for m in at.markdown] + [c.value for c in at.caption]


def test_tabs_help_center_and_footer_without_running_experiment(no_network):
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Experiment", "Help Center"]
    assert [e.label for e in at.expander] == HELP_SECTIONS
    assert "report" not in at.session_state  # loading the page / help runs nothing
    texts = " ".join(_texts(at))
    assert "https://rikiyat.github.io/blog/weight-space-programming/" in texts
    assert "https://www.youtube.com/watch?v=VAbXapGNveU" in texts
    assert CREDIT in [c.value for c in at.caption]


def _pyarrow_loads() -> bool:
    try:
        import pyarrow  # noqa: F401  (st.dataframe needs it)
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _pyarrow_loads(), reason="pyarrow cannot be loaded on this host (needed by st.dataframe)")
def test_result_survives_reruns_after_running(no_network):
    at =AppTest.from_file(APP, default_timeout=60).run()
    at.button[0].click().run()
    assert at.session_state["report"]["status"] == "accepted"
    first = at.session_state["report"]["timestamp"], at.session_state["run_count"]
    at.run()  # any later interaction reruns the script; the last result must stay
    assert (at.session_state["report"]["timestamp"], at.session_state["run_count"]) == first
    assert any("ACCEPTED" in s.value for s in at.success)
    assert [e.label for e in at.expander][-5:] == HELP_SECTIONS
