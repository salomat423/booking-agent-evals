"""Talk to the agent in the terminal.  python chat.py

Each turn prints the tool calls the agent made and where its trace was written.
"""
from agent import Agent, BookingStore, KnowledgeBase
from agent.tools import Toolbox
from agent.trace import Tracer


def main() -> None:
    toolbox = Toolbox(BookingStore.from_json(), KnowledgeBase.from_markdown())
    agent = Agent(toolbox)
    tracer = Tracer()
    history: list = []
    print(f"Demo Barbershop assistant (today is {toolbox.store.today}). Empty line to quit.\n")
    while True:
        user = input("you> ").strip()
        if not user:
            break
        history.append({"role": "user", "content": user})
        result = agent.run(history, tracer)
        history = result.messages
        for call in result.tool_calls[-10:]:
            print(f"  [{call['status']}] {call['name']} {call['input']}")
        print(f"bot> {result.text}\n")
    print(f"trace: {tracer.path}")


if __name__ == "__main__":
    main()
