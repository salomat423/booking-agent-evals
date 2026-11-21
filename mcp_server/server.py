"""MCP server exposing the barbershop's booking tools.

Any MCP client (Claude Desktop, Claude Code, an IDE) can use the same tools the agent
uses, through the same validation layer. Run:  python -m mcp_server.server  (stdio)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.mcpserver import MCPServer  # noqa: E402

from agent.knowledge import KnowledgeBase  # noqa: E402
from agent.store import BookingStore  # noqa: E402
from agent.tools import Toolbox  # noqa: E402

toolbox = Toolbox(BookingStore.from_json(), KnowledgeBase.from_markdown())

server = MCPServer(
    name="demo-barbershop",
    instructions=(
        "Booking tools for Demo Barbershop (Astana). Look facts up with search_knowledge, "
        "check free slots before booking, and only book after the client confirms."
    ),
)


def _result(name: str, args: dict) -> dict:
    outcome = toolbox.run(name, args)
    return {"status": outcome.status, **outcome.content}


@server.tool()
def search_knowledge(query: str) -> dict:
    """Search services, prices, hours, masters and policies of the barbershop."""
    return _result("search_knowledge", {"query": query})


@server.tool()
def check_availability(date: str, service_id: str, master: str = "any") -> dict:
    """Free slots for a service on a date (YYYY-MM-DD). service_id: haircut|beard|combo|kids|shave.
    master: Arman|Dias|any."""
    return _result("check_availability", {"date": date, "service_id": service_id, "master": master})


@server.tool()
def create_booking(service_id: str, date: str, time: str, master: str, client_name: str,
                   phone: str, client_confirmed: bool) -> dict:
    """Create a booking. Rejected with reasons if the slot is taken, data is invalid or the
    client has not confirmed."""
    return _result("create_booking", {
        "service_id": service_id, "date": date, "time": time, "master": master,
        "client_name": client_name, "phone": phone, "client_confirmed": client_confirmed,
    })


@server.tool()
def handoff_to_human(reason: str, summary: str) -> dict:
    """Pass the conversation to the manager. reason: complaint|refund|health|out_of_scope|client_request."""
    return _result("handoff_to_human", {"reason": reason, "summary": summary})


if __name__ == "__main__":
    server.run("stdio")
