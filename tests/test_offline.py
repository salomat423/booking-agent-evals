"""Offline tests: no API key, no network. They cover retrieval, guardrails, the agent loop
(with a scripted fake model), the deterministic eval checks and the MCP server wiring."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from agent import Agent, BookingStore, KnowledgeBase
from agent.guardrails import normalize_phone, validate_booking
from agent.tools import TOOL_SCHEMAS, Toolbox
from agent.trace import Tracer
from evals.checks import run_checks
from evals.run_evals import load_cases


@pytest.fixture
def store():
    return BookingStore.from_json()


@pytest.fixture
def kb():
    return KnowledgeBase.from_markdown()


GOOD = {"service_id": "haircut", "date": "2026-11-24", "time": "17:00", "master": "Arman",
        "client_name": "Timur", "phone": "+7 701 234 5678", "client_confirmed": True}


# ---- retrieval --------------------------------------------------------------------
@pytest.mark.parametrize("query, section", [
    ("price of a men's haircut", "Services and prices"),
    ("cancel appointment", "Cancellation and lateness"),
    ("skin irritation refund", "Complaints, refunds and health issues"),
    ("opening hours address", "Opening hours and address"),
])
def test_retrieval_finds_right_section(kb, query, section):
    assert kb.search(query, k=2)[0].title == section


# ---- guardrails -------------------------------------------------------------------
def test_valid_booking_passes(store):
    assert validate_booking(store, GOOD).ok


@pytest.mark.parametrize("change, problem", [
    ({"phone": ""}, "phone"),
    ({"client_name": " "}, "client_name"),
    ({"time": "15:00"}, "already booked"),
    ({"date": "2026-11-23"}, "outside the bookable range"),
    ({"master": "Dias", "service_id": "combo"}, "does not do"),
    ({"client_confirmed": False}, "confirmed"),
    ({"time": "21:00"}, "not a valid slot"),
])
def test_invalid_booking_is_blocked(store, change, problem):
    decision = validate_booking(store, {**GOOD, **change})
    assert not decision.ok
    assert any(problem in p for p in decision.problems), decision.problems


def test_phone_normalization():
    assert normalize_phone("8 (701) 234-56-78") == "+77012345678"
    assert normalize_phone("+7 701 234 5678") == "+77012345678"


def test_tool_schemas_are_strict():
    for tool in TOOL_SCHEMAS:
        schema = tool["input_schema"]
        assert tool["strict"] and schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])


# ---- agent loop with a scripted model ---------------------------------------------
def _resp(stop_reason, *blocks):
    return SimpleNamespace(stop_reason=stop_reason, content=list(blocks), model="fake",
                           usage=SimpleNamespace(input_tokens=100, output_tokens=20,
                                                 cache_read_input_tokens=0))


def _tool(id_, name, **inp):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=inp)


def _text(t):
    return SimpleNamespace(type="text", text=t)


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


def test_loop_blocks_bad_booking_then_books(store, kb, tmp_path):
    client = FakeClient([
        _resp("tool_use", _tool("t1", "create_booking", **{**GOOD, "time": "15:00"})),
        _resp("tool_use", _tool("t2", "create_booking", **GOOD)),
        _resp("end_turn", _text("Booked: haircut with Arman, 24 Nov at 17:00.")),
    ])
    toolbox = Toolbox(store, kb)
    result = Agent(toolbox, client).run([{"role": "user", "content": "book"}], Tracer(tmp_path))

    assert result.status == "ok" and result.steps == 3
    assert [c["status"] for c in result.tool_calls] == ["blocked", "ok"]
    assert len(store.bookings) == 1 and store.bookings[0]["phone"] == "+77012345678"
    # the rejection reached the model as an error tool_result
    # messages: [user, assistant(tool_use t1), user(tool_result t1), ...]
    blocked = client.requests[1]["messages"][2]["content"][0]
    assert blocked["is_error"] is True and "already booked" in blocked["content"]
    # every step is in the trace
    lines = (tmp_path / f"{result.run_id}.jsonl").read_text(encoding="utf-8").splitlines()
    events = [json.loads(line)["event"] for line in lines]
    assert events.count("tool_call") == 2 and events[-1] == "final"


def test_loop_stops_at_step_limit(store, kb, tmp_path):
    client = FakeClient([_resp("tool_use", _tool(f"t{i}", "search_knowledge", query="x")) for i in range(3)])
    toolbox = Toolbox(store, kb)
    result = Agent(toolbox, client, max_steps=3).run([{"role": "user", "content": "?"}], Tracer(tmp_path))
    assert result.status == "max_steps" and toolbox.handoffs


def test_loop_handles_refusal(store, kb, tmp_path):
    client = FakeClient([_resp("refusal")])
    result = Agent(Toolbox(store, kb), client).run([{"role": "user", "content": "?"}], Tracer(tmp_path))
    assert result.status == "refusal" and result.text


# ---- eval harness -----------------------------------------------------------------
def test_golden_set_is_well_formed():
    cases = load_cases()
    assert len(cases) >= 10 and len({c["id"] for c in cases}) == len(cases)
    for c in cases:
        assert c["conversation"][-1]["role"] == "user" and c["rubric"] and c["expect"]


def test_checks_catch_wrong_booking():
    from agent.loop import AgentResult
    result = AgentResult(text="Done!", status="ok", tool_calls=[
        {"name": "create_booking", "input": {**GOOD, "time": "16:00"}, "status": "ok", "content": {}}])
    expect = {"tool_args": {"create_booking": {"time": "17:00"}}, "tools_not_called_ok": ["create_booking"]}
    failed = {c.name for c in run_checks(expect, result) if not c.passed}
    assert failed == {"args:create_booking", "no_successful:create_booking"}


# ---- MCP server -------------------------------------------------------------------
def test_mcp_server_exposes_tools():
    from mcp_server.server import server
    tools = asyncio.run(server.list_tools())
    assert {t.name for t in tools} == {"search_knowledge", "check_availability", "create_booking",
                                       "handoff_to_human"}
