"""Deterministic checks: fast, free, no model involved.

They verify behaviour that has one right answer: which tools were called, with which
arguments, whether a booking was (not) created, and hard strings that must or must not
appear in the reply. Anything that needs judgement goes to the LLM judge instead.
"""
from __future__ import annotations

from dataclasses import dataclass

from agent.guardrails import normalize_phone
from agent.loop import AgentResult


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


def _norm(value):
    return normalize_phone(value) if isinstance(value, str) and value.startswith("+") else value


def run_checks(expect: dict, result: AgentResult) -> list[Check]:
    checks: list[Check] = []
    text = result.text.lower()
    called = [c["name"] for c in result.tool_calls]

    checks.append(Check("finished", result.status == "ok", f"status={result.status}"))

    for name in expect.get("tools_called", []):
        checks.append(Check(f"called:{name}", name in called, f"calls={called}"))

    for name in expect.get("tools_not_called_ok", []):
        ok_calls = result.called(name, "ok")
        checks.append(Check(f"no_successful:{name}", not ok_calls,
                            f"{len(ok_calls)} successful call(s)" if ok_calls else ""))

    for name, wanted in expect.get("tool_args", {}).items():
        calls = [c for c in result.tool_calls if c["name"] == name]
        match = any(all(_norm(c["input"].get(k)) == _norm(v) for k, v in wanted.items()) for c in calls)
        checks.append(Check(f"args:{name}", match,
                            f"wanted {wanted}; got {[c['input'] for c in calls]}"))

    if "booking_ok" in expect:
        ok = bool(result.called("create_booking", "ok"))
        checks.append(Check("booking_created", ok == expect["booking_ok"],
                            f"created={ok}"))

    if expect.get("answer_contains_any"):
        options = expect["answer_contains_any"]
        checks.append(Check("answer_contains_any", any(o.lower() in text for o in options),
                            f"none of {options}"))

    for bad in expect.get("answer_not_contains", []):
        checks.append(Check(f"answer_not_contains:{bad}", bad.lower() not in text))

    return checks
