"""Run the golden set against the agent and report a score.

    python -m evals.run_evals                  # all cases, deterministic checks + LLM judge
    python -m evals.run_evals --no-judge       # deterministic checks only (cheaper)
    python -m evals.run_evals --case booking_happy_path --repeat 3
    python -m evals.run_evals --threshold 0.9  # exit code 1 below 90% (for CI)

A case passes when all its deterministic checks pass and (if enabled) the judge passes.
Every case runs on a fresh copy of the calendar, and every agent run leaves a trace.
Results are saved to evals/results/<timestamp>.json so runs can be compared over time.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import Agent, BookingStore, KnowledgeBase, config  # noqa: E402
from agent.tools import Toolbox  # noqa: E402
from agent.trace import Tracer  # noqa: E402
from evals.checks import run_checks  # noqa: E402
from evals.judge import judge  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "golden.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_cases(path: Path = GOLDEN) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_case(case: dict, *, client, base_store: BookingStore, kb: KnowledgeBase, kb_text: str,
             use_judge: bool, trace_dir: Path) -> dict:
    toolbox = Toolbox(base_store.fresh_copy(), kb)
    agent = Agent(toolbox, client)
    tracer = Tracer(trace_dir, run_id=f"{case['id']}-{time.strftime('%H%M%S')}")
    result = agent.run(case["conversation"], tracer)

    checks = run_checks(case.get("expect", {}), result)
    det_pass = all(c.passed for c in checks)
    verdict = None
    if use_judge and case.get("rubric"):
        verdict = judge(client, kb_text=kb_text, conversation=case["conversation"],
                        result=result, rubric=case["rubric"])
    passed = det_pass and (verdict is None or verdict.get("pass") is True)
    return {
        "id": case["id"], "category": case.get("category", ""), "passed": passed,
        "deterministic_pass": det_pass, "judge": verdict,
        "failed_checks": [{"name": c.name, "detail": c.detail} for c in checks if not c.passed],
        "reply": result.text, "tool_calls": [{k: c[k] for k in ("name", "input", "status")}
                                            for c in result.tool_calls],
        "status": result.status, "steps": result.steps, "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens, "latency_ms": result.latency_ms,
        "trace": str(tracer.path) if tracer.path else None,
    }


def summarize(rows: list[dict]) -> dict:
    by_cat: dict[str, list[bool]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r["passed"])
    return {
        "cases": len(rows),
        "pass_rate": round(sum(r["passed"] for r in rows) / len(rows), 3),
        "deterministic_pass_rate": round(sum(r["deterministic_pass"] for r in rows) / len(rows), 3),
        "by_category": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_cat.items())},
        "avg_input_tokens": int(statistics.mean(r["input_tokens"] for r in rows)),
        "avg_output_tokens": int(statistics.mean(r["output_tokens"] for r in rows)),
        "p50_latency_ms": int(statistics.median(r["latency_ms"] for r in rows)),
        "max_latency_ms": max(r["latency_ms"] for r in rows),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", action="append", help="run only this case id (repeatable)")
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM judge")
    ap.add_argument("--repeat", type=int, default=1, help="run each case N times to see flakiness")
    ap.add_argument("--threshold", type=float, default=0.0, help="fail (exit 1) below this pass rate")
    args = ap.parse_args(argv)

    cases = load_cases()
    if args.case:
        cases = [c for c in cases if c["id"] in set(args.case)]
        if not cases:
            print(f"no cases match {args.case}")
            return 2

    client = anthropic.Anthropic()
    base_store = BookingStore.from_json()
    kb = KnowledgeBase.from_markdown()
    kb_text = (config.DATA_DIR / "knowledge_base.md").read_text(encoding="utf-8")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    trace_dir = config.TRACES_DIR / f"eval-{stamp}"

    print(f"model={config.AGENT_MODEL} effort={config.AGENT_EFFORT} "
          f"judge={'off' if args.no_judge else config.JUDGE_MODEL} cases={len(cases)} x{args.repeat}\n")
    rows = []
    for case in cases:
        for i in range(args.repeat):
            row = run_case(case, client=client, base_store=base_store, kb=kb, kb_text=kb_text,
                           use_judge=not args.no_judge, trace_dir=trace_dir)
            row["attempt"] = i + 1
            rows.append(row)
            mark = "PASS" if row["passed"] else "FAIL"
            print(f"{mark}  {row['id']:<32} steps={row['steps']} tokens={row['input_tokens']}+"
                  f"{row['output_tokens']} {row['latency_ms']}ms")
            for fc in row["failed_checks"]:
                print(f"      x {fc['name']}: {fc['detail']}")
            if row["judge"] and not row["judge"].get("pass"):
                print(f"      x judge: {row['judge'].get('reason')}")

    summary = summarize(rows)
    print("\n" + json.dumps(summary, indent=2))
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{stamp}.json"
    out.write_text(json.dumps({"config": {"agent_model": config.AGENT_MODEL, "effort": config.AGENT_EFFORT,
                                          "judge_model": None if args.no_judge else config.JUDGE_MODEL},
                               "summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\nsaved {out}")
    return 0 if summary["pass_rate"] >= args.threshold else 1


if __name__ == "__main__":
    sys.exit(main())
