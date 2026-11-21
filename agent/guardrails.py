"""Validation layer: the model proposes an action, the application decides.

Every tool call that changes state goes through a check here before it runs. A failed
check is returned to the model as an error it can recover from ("that slot is taken,
here is why"), and it is recorded in the trace, so blocked actions are visible in evals.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from .store import BookingStore

PHONE_RE = re.compile(r"^\+7\d{10}$")


@dataclass
class Decision:
    ok: bool
    problems: list[str]

    @classmethod
    def allow(cls) -> "Decision":
        return cls(True, [])


def normalize_phone(phone: str) -> str:
    digits = re.sub(r"\D", "", phone or "")
    if len(digits) == 11 and digits[0] in "78":
        return "+7" + digits[1:]
    if len(digits) == 10:
        return "+7" + digits
    return phone or ""


def check_date(store: BookingStore, value: str) -> tuple[date | None, list[str]]:
    try:
        day = date.fromisoformat(value)
    except (TypeError, ValueError):
        return None, [f"date '{value}' is not in YYYY-MM-DD format"]
    first, last = store.bookable_dates()
    if not first <= day <= last:
        return day, [f"date {day} is outside the bookable range {first}..{last}"]
    return day, []


def validate_booking(store: BookingStore, args: dict) -> Decision:
    problems: list[str] = []
    service_id = args.get("service_id")
    if service_id not in store.services:
        problems.append(f"unknown service_id '{service_id}'; valid: {sorted(store.services)}")

    day, date_problems = check_date(store, args.get("date", ""))
    problems += date_problems

    time = args.get("time", "")
    if time not in store.hours():
        problems.append(f"time '{time}' is not a valid slot; slots start on the hour "
                        f"{store.hours()[0]}..{store.hours()[-1]}")

    master = args.get("master")
    if service_id in store.services and master not in store.services[service_id]["masters"]:
        problems.append(f"master '{master}' does not do '{service_id}'; "
                        f"valid: {store.services[service_id]['masters']}")

    if not (args.get("client_name") or "").strip():
        problems.append("client_name is missing: ask the client for their name")

    phone = normalize_phone(args.get("phone", ""))
    if not PHONE_RE.match(phone):
        problems.append("phone is missing or not a Kazakhstan mobile number (+7XXXXXXXXXX): "
                        "ask the client for it")

    if not problems and (day.isoformat(), time, master) in store.taken:
        problems.append(f"{master} is already booked on {day} at {time}; "
                        "call check_availability and offer free slots instead")

    if args.get("client_confirmed") is not True:
        problems.append("the client has not explicitly confirmed these details; "
                        "repeat them and ask for confirmation first")

    return Decision(not problems, problems)
