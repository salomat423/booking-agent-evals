"""Tool definitions and their implementations.

The same `Toolbox` is used by the agent loop and by the MCP server, so both paths share
one set of business rules (see guardrails.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .guardrails import check_date, normalize_phone, validate_booking
from .knowledge import KnowledgeBase
from .store import BookingStore

SERVICE_IDS = ["haircut", "beard", "combo", "kids", "shave"]
MASTERS = ["Arman", "Dias"]

TOOL_SCHEMAS = [
    {
        "name": "search_knowledge",
        "description": (
            "Search the barbershop knowledge base: services, prices, discounts, opening hours, "
            "address, masters, booking rules, cancellation, payment and complaint policies. "
            "Use it before stating any fact about the business; never answer facts from memory."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to look up, in English."}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "check_availability",
        "description": (
            "List free appointment slots for a service on a given date. "
            "Call it before suggesting or booking any time."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "YYYY-MM-DD"},
                "service_id": {"type": "string", "enum": SERVICE_IDS},
                "master": {"type": "string", "enum": MASTERS + ["any"]},
            },
            "required": ["date", "service_id", "master"],
            "additionalProperties": False,
        },
    },
    {
        "name": "create_booking",
        "description": (
            "Create an appointment. Only call it after the client has explicitly confirmed the "
            "service, date, time, master, their name and phone number. The application validates "
            "the request and may reject it with a reason."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "service_id": {"type": "string", "enum": SERVICE_IDS},
                "date": {"type": "string", "description": "YYYY-MM-DD"},
                "time": {"type": "string", "description": "HH:00, e.g. 14:00"},
                "master": {"type": "string", "enum": MASTERS},
                "client_name": {"type": "string"},
                "phone": {"type": "string", "description": "Kazakhstan mobile, +7XXXXXXXXXX"},
                "client_confirmed": {
                    "type": "boolean",
                    "description": "True only if the client said yes to exactly these details.",
                },
            },
            "required": ["service_id", "date", "time", "master", "client_name", "phone",
                         "client_confirmed"],
            "additionalProperties": False,
        },
    },
    {
        "name": "handoff_to_human",
        "description": (
            "Pass the conversation to the manager. Use it for complaints, refunds, health issues, "
            "requests outside the knowledge base, or when the client asks for a person."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "enum": ["complaint", "refund", "health", "out_of_scope",
                                                      "client_request"]},
                "summary": {"type": "string", "description": "One or two sentences for the manager."},
            },
            "required": ["reason", "summary"],
            "additionalProperties": False,
        },
    },
]


@dataclass
class ToolOutcome:
    status: str  # "ok" | "blocked" | "error"
    content: dict

    @property
    def is_error(self) -> bool:
        return self.status != "ok"


@dataclass
class Toolbox:
    store: BookingStore
    kb: KnowledgeBase
    handoffs: list = field(default_factory=list)

    def run(self, name: str, args: dict) -> ToolOutcome:
        handler = getattr(self, f"_{name}", None)
        if handler is None:
            return ToolOutcome("error", {"error": f"unknown tool '{name}'"})
        try:
            return handler(args)
        except Exception as exc:  # a tool bug must not crash the conversation
            return ToolOutcome("error", {"error": f"{type(exc).__name__}: {exc}"})

    def _search_knowledge(self, args: dict) -> ToolOutcome:
        hits = self.kb.search(args["query"], k=2)
        if not hits:
            return ToolOutcome("ok", {"results": [], "note": "Nothing found. Do not guess; "
                                                              "say you don't know or hand off."})
        return ToolOutcome("ok", {"results": [{"section": h.title, "text": h.text} for h in hits]})

    def _check_availability(self, args: dict) -> ToolOutcome:
        day, problems = check_date(self.store, args["date"])
        if args["service_id"] not in self.store.services:
            problems.append(f"unknown service_id '{args['service_id']}'")
        if problems:
            return ToolOutcome("blocked", {"error": "; ".join(problems)})
        master = None if args.get("master") in (None, "any") else args["master"]
        slots = self.store.free_slots(day, args["service_id"], master)
        return ToolOutcome("ok", {"date": day.isoformat(), "service_id": args["service_id"],
                                  "free_slots": slots})

    def _create_booking(self, args: dict) -> ToolOutcome:
        decision = validate_booking(self.store, args)
        if not decision.ok:
            return ToolOutcome("blocked", {"error": "booking rejected by validation",
                                           "problems": decision.problems})
        booking = self.store.book(
            service_id=args["service_id"], day=check_date(self.store, args["date"])[0],
            time=args["time"], master=args["master"],
            client_name=args["client_name"].strip(), phone=normalize_phone(args["phone"]),
        )
        return ToolOutcome("ok", {"booking": booking})

    def _handoff_to_human(self, args: dict) -> ToolOutcome:
        ticket = {"ticket_id": f"H{len(self.handoffs) + 1:04d}", **args}
        self.handoffs.append(ticket)
        return ToolOutcome("ok", {"ticket": ticket,
                                  "note": "Tell the client a manager will contact them soon."})
