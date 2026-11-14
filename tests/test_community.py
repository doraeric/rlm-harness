"""`rlm_harness.community`: the routing table for LMs on a vendor's consumer sign-in, its warning,
its document, and the alias that keeps the old `rlm_harness.claude_agent_lm` path working."""

import inspect
import logging
import re
import subprocess
import sys
import threading
import types
from pathlib import Path

import pytest

from rlm_harness import community
from rlm_harness.community import EXPLICIT, NOT_EXPLICIT, ROUTES, Route, route_for

dspy = pytest.importorskip("dspy")

_README = Path(community.__file__).with_name("README.md")


def test_importing_community_does_not_import_dspy():
    code = (
        "import sys, rlm_harness.community; "
        "assert 'dspy' not in sys.modules, 'rlm_harness.community pulled in dspy'; print('ok')"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_the_old_module_path_is_the_moved_module():
    import rlm_harness.claude_agent_lm as old
    import rlm_harness.community.claude_agent_lm as new
    from rlm_harness.claude_agent_lm import ClaudeAgentLM

    assert old is new
    assert ClaudeAgentLM is new.ClaudeAgentLM
    assert community.ClaudeAgentLM is new.ClaudeAgentLM


@pytest.mark.parametrize("route", ROUTES, ids=lambda r: r.prefix)
def test_every_route_names_an_lm_class_that_takes_a_deadline(route):
    """Checked on the class, not an instance: building one needs the vendor's tool and sign-in."""
    import importlib

    cls = getattr(importlib.import_module(route.module), route.attr)
    assert issubclass(cls, dspy.LM)
    assert "timeout_s" in inspect.signature(cls.__init__).parameters
    assert route.vendor_permission in (EXPLICIT, NOT_EXPLICIT)
    assert route.prefix.endswith("/") and route.example


def test_no_prefix_shadows_another_or_a_litellm_provider():
    prefixes = [r.prefix for r in ROUTES]
    for a in prefixes:
        for b in prefixes:
            assert a == b or not b.startswith(a), f"{a!r} would route {b!r} model strings"
    litellm = pytest.importorskip("litellm")
    providers = {str(getattr(p, "value", p)) for p in litellm.provider_list}
    taken = [p for p in prefixes if p[:-1] in providers]
    assert not taken, f"{taken} would capture model strings litellm routes by API key"


@pytest.mark.parametrize("route", ROUTES, ids=lambda r: r.prefix)
def test_every_shipped_prefix_routes_and_names_its_example(route, monkeypatch):
    """Each real prefix, not only a stub: a model string under it routes to its class, and a bare
    prefix raises naming the route's own example."""
    import importlib

    import rlm_harness.runtime as rt

    module = importlib.import_module(route.module)
    built = []
    monkeypatch.setattr(module, route.attr, lambda model_id: built.append(model_id) or object())
    assert route_for(route.prefix + "x") is route
    _lm, matched = rt._maybe_subscription_lm(route.prefix + route.example)
    assert matched is route and built == [route.example]
    with pytest.raises(ValueError, match=re.escape(f"e.g. {route.prefix}{route.example}.")):
        rt._maybe_subscription_lm(route.prefix)


def test_route_for_matches_by_prefix_only():
    assert route_for("claude-agent-sdk/sonnet").attr == "ClaudeAgentLM"
    assert route_for("anthropic/claude-sonnet-5") is None
    assert route_for("sonnet") is None


def test_the_document_has_a_dated_row_with_a_link_for_every_route():
    text = _README.read_text(encoding="utf-8")
    for route in ROUTES:
        rows = [line for line in text.splitlines() if line.startswith("|") and f"`{route.prefix}" in line]
        assert len(rows) == 1, f"{route.prefix}: expected one table row, found {len(rows)}"
        assert re.search(r"\]\(https?://|https?://", rows[0]), f"{route.prefix}: no link"
        assert re.search(r"\b20\d\d-\d\d-\d\d\b", rows[0]), f"{route.prefix}: no date checked"


# --- the warning -----------------------------------------------------------------------------

def _stub_route(monkeypatch, permission):
    module = types.ModuleType("rlm_harness_test_stub_route")

    class StubLM(dspy.LM):
        def __init__(self, model, *, timeout_s=1.0):
            community._note_constructed("StubLM")
            super().__init__(f"stub/{model}")

    module.StubLM = StubLM
    monkeypatch.setitem(sys.modules, module.__name__, module)
    route = Route("stub/", module.__name__, "StubLM", "m", permission)
    monkeypatch.setattr(community, "ROUTES", (*ROUTES, route))
    monkeypatch.setattr(community, "_noted", set())
    return StubLM


def _warnings(caplog):
    return [r for r in caplog.records if r.name == "rlm_harness.community" and r.levelno == logging.WARNING]


def test_a_route_not_explicitly_allowed_warns_once_per_process(monkeypatch, caplog):
    stub = _stub_route(monkeypatch, NOT_EXPLICIT)
    with caplog.at_level(logging.WARNING, logger="rlm_harness.community"):
        threads = [threading.Thread(target=stub, args=("m",)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        stub("m")
    found = _warnings(caplog)
    assert len(found) == 1
    assert "'stub/'" in found[0].getMessage() and "community/README.md" in found[0].getMessage()


def test_an_explicitly_allowed_route_says_nothing(monkeypatch, caplog):
    stub = _stub_route(monkeypatch, EXPLICIT)
    with caplog.at_level(logging.WARNING, logger="rlm_harness.community"):
        stub("m")
    assert _warnings(caplog) == []


def test_configure_routes_a_new_prefix_with_no_change_to_runtime(monkeypatch):
    """Adding a route is one `Route` entry: routing, the bare-prefix error and the class all come
    from it."""
    import rlm_harness.runtime as rt

    stub = _stub_route(monkeypatch, EXPLICIT)
    lm, route = rt._maybe_subscription_lm("stub/m")
    assert isinstance(lm, stub) and route.attr == "StubLM"
    with pytest.raises(ValueError, match=r"expected stub/<id>, e\.g\. stub/m\."):
        rt._maybe_subscription_lm("stub/")
    assert rt._maybe_subscription_lm("other/m") is None
