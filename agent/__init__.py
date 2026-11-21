"""Booking agent: an LLM agent that answers questions and books appointments for a barbershop.

The model proposes actions; the application validates and executes them.
"""
from .loop import Agent, AgentResult
from .knowledge import KnowledgeBase
from .store import BookingStore

__all__ = ["Agent", "AgentResult", "KnowledgeBase", "BookingStore"]
