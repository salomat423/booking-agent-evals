"""The agent loop.

A manual loop on the Messages API instead of a ready-made runner, on purpose: between
"the model asked for a tool" and "the tool ran" sits the application's validation layer,
and every step is written to a trace. Both are the point of this project.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import anthropic

from . import config
from .prompts import build_system_prompt
from .tools import TOOL_SCHEMAS, Toolbox
from .trace import Tracer


@dataclass
class AgentResult:
    text: str
    status: str  # ok | refusal | max_tokens | max_steps | api_error
    tool_calls: list[dict] = field(default_factory=list)
    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    run_id: str = ""
    messages: list = field(default_factory=list)

    def called(self, name: str, status: str | None = None) -> list[dict]:
        return [c for c in self.tool_calls if c["name"] == name and (status is None or c["status"] == status)]


def _text_of(content) -> str:
    return "\n".join(b.text for b in content if getattr(b, "type", None) == "text").strip()


class Agent:
    def __init__(self, toolbox: Toolbox, client=None, *, model: str = config.AGENT_MODEL,
                 effort: str = config.AGENT_EFFORT, max_steps: int = config.MAX_STEPS,
                 use_fallbacks: bool = config.USE_FALLBACKS):
        self.toolbox = toolbox
        self.client = client or anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.max_steps = max_steps
        self.use_fallbacks = use_fallbacks

    def _request(self, messages: list) -> object:
        kwargs = dict(
            model=self.model,
            max_tokens=config.MAX_TOKENS,
            system=build_system_prompt(self.toolbox.store.today),
            tools=TOOL_SCHEMAS,
            messages=messages,
            output_config={"effort": self.effort},
        )
        if self.use_fallbacks:
            # If the model declines, the API re-runs the request on a fallback model in the same call.
            kwargs.update(betas=[config.FALLBACK_BETA], fallbacks="default")
        return self.client.beta.messages.create(**kwargs)

    def run(self, conversation: list[dict], tracer: Tracer | None = None) -> AgentResult:
        """Continue a conversation (list of {"role", "content"} turns ending with the user)."""
        tracer = tracer or Tracer()
        messages = [dict(m) for m in conversation]
        result = AgentResult(text="", status="ok", run_id=tracer.run_id)
        tracer.log("start", model=self.model, effort=self.effort,
                   last_user_message=str(messages[-1]["content"])[:500])
        started = time.perf_counter()

        for step in range(1, self.max_steps + 1):
            result.steps = step
            t0 = time.perf_counter()
            try:
                response = self._request(messages)
            except anthropic.APIError as exc:
                tracer.log("api_error", step=step, error=f"{type(exc).__name__}: {exc}")
                result.status = "api_error"
                result.text = "Sorry, something went wrong on our side. A manager will reply to you shortly."
                break
            usage = response.usage
            result.input_tokens += usage.input_tokens or 0
            result.output_tokens += usage.output_tokens or 0
            tracer.log("model_response", step=step, stop_reason=response.stop_reason,
                       input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
                       cache_read_tokens=getattr(usage, "cache_read_input_tokens", None),
                       latency_ms=int((time.perf_counter() - t0) * 1000), model=response.model,
                       text=_text_of(response.content)[:500])

            # Keep the full content (thinking blocks included) so the next request is valid.
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "pause_turn":
                continue
            if response.stop_reason == "refusal":
                result.status = "refusal"
                result.text = "I can't help with that here. A manager will contact you."
                break
            if response.stop_reason == "max_tokens":
                result.status = "max_tokens"
                result.text = _text_of(response.content)
                break
            if response.stop_reason != "tool_use":
                result.text = _text_of(response.content)
                break

            tool_results = []
            for block in response.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                tracer.log("tool_call", step=step, name=block.name, input=block.input)
                outcome = self.toolbox.run(block.name, dict(block.input))
                tracer.log("tool_result", step=step, name=block.name, status=outcome.status,
                           content=outcome.content)
                result.tool_calls.append({"name": block.name, "input": dict(block.input),
                                          "status": outcome.status, "content": outcome.content})
                tool_results.append({"type": "tool_result", "tool_use_id": block.id,
                                     "content": _json(outcome.content), "is_error": outcome.is_error})
            # All results of one turn go back in a single user message.
            messages.append({"role": "user", "content": tool_results})
        else:
            result.status = "max_steps"
            result.text = "Let me pass you to a manager so we get this right."
            self.toolbox.run("handoff_to_human", {"reason": "out_of_scope",
                                                  "summary": "Agent hit the step limit."})

        result.latency_ms = int((time.perf_counter() - started) * 1000)
        result.messages = messages
        tracer.log("final", status=result.status, steps=result.steps, text=result.text,
                   total_input_tokens=result.input_tokens, total_output_tokens=result.output_tokens,
                   latency_ms=result.latency_ms)
        return result


def _json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)
