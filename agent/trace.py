"""Observability: one JSONL file per conversation, one line per event.

Each line records what the model saw and did at one step: request size, stop reason,
tokens, latency, every tool call with its input, the validation decision and the result.
`python -m agent.trace traces/<file>.jsonl` prints a readable timeline.
"""
from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

from .config import TRACES_DIR


class Tracer:
    def __init__(self, directory: Path | None = TRACES_DIR, run_id: str | None = None):
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.events: list[dict] = []
        self.path = None
        if directory is not None:
            Path(directory).mkdir(parents=True, exist_ok=True)
            self.path = Path(directory) / f"{self.run_id}.jsonl"

    def log(self, event: str, **data) -> None:
        record = {"ts": round(time.time(), 3), "run_id": self.run_id, "event": event, **data}
        self.events.append(record)
        if self.path:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def print_timeline(path: str) -> None:
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        kind = e["event"]
        if kind == "model_response":
            print(f"step {e['step']}: model -> {e['stop_reason']}  "
                  f"in={e['input_tokens']} out={e['output_tokens']} {e['latency_ms']}ms")
        elif kind == "tool_call":
            print(f"   tool {e['name']}({json.dumps(e['input'], ensure_ascii=False)})")
        elif kind == "tool_result":
            print(f"   -> {e['status']}: {json.dumps(e['content'], ensure_ascii=False)[:160]}")
        elif kind == "final":
            print(f"final ({e['steps']} steps, {e['total_input_tokens']}+{e['total_output_tokens']} "
                  f"tokens): {e['text'][:200]}")
        else:
            print(f"{kind}: {json.dumps({k: v for k, v in e.items() if k not in ('ts', 'run_id', 'event')}, ensure_ascii=False)[:200]}")


if __name__ == "__main__":
    print_timeline(sys.argv[1])
