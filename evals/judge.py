"""LLM-as-judge for the parts of a reply that need judgement (tone, honesty, completeness).

The judge gets the knowledge base as ground truth, the conversation, the tool calls and
the final reply, plus a per-case rubric. It must return {"pass": bool, "reason": str}
through structured output, so the result is always parseable.
"""
from __future__ import annotations

import json

import anthropic

from agent import config
from agent.loop import AgentResult

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string", "description": "One or two sentences explaining the verdict."},
        "pass": {"type": "boolean"},
    },
    "required": ["reason", "pass"],
    "additionalProperties": False,
}

JUDGE_PROMPT = """\
You grade one reply of a barbershop booking assistant.

<knowledge_base>
{kb}
</knowledge_base>

<conversation>
{conversation}
</conversation>

<tool_calls>
{tool_calls}
</tool_calls>

<final_reply>
{reply}
</final_reply>

<rubric>
{rubric}
</rubric>

Pass the reply only if it satisfies every point of the rubric and states no fact that
contradicts the knowledge base. Minor wording issues are fine. Explain briefly, then decide."""


def judge(client: anthropic.Anthropic, *, kb_text: str, conversation: list[dict],
          result: AgentResult, rubric: str) -> dict:
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in conversation)
    calls = json.dumps([{k: c[k] for k in ("name", "input", "status")} for c in result.tool_calls],
                       ensure_ascii=False, indent=1)
    response = client.beta.messages.create(
        model=config.JUDGE_MODEL,
        max_tokens=2048,
        betas=[config.FALLBACK_BETA],
        fallbacks="default",
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            kb=kb_text, conversation=convo, tool_calls=calls, reply=result.text, rubric=rubric)}],
    )
    if response.stop_reason == "refusal":
        return {"pass": False, "reason": "judge refused", "error": True}
    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        verdict = json.loads(text)
    except json.JSONDecodeError:
        return {"pass": False, "reason": f"unparseable judge output: {text[:200]}", "error": True}
    verdict["input_tokens"] = response.usage.input_tokens
    verdict["output_tokens"] = response.usage.output_tokens
    return verdict
