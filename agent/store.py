"""In-memory booking calendar loaded from data/schedule.json.

In production this was a CRM behind an API; here it is a small deterministic store so
evals can run offline and every run starts from the same state.
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from .config import DATA_DIR


@dataclass
class BookingStore:
    today: date
    opening_hour: int
    last_start_hour: int
    max_days_ahead: int
    services: dict
    taken: set = field(default_factory=set)  # {(date_iso, "HH:00", master)}
    bookings: list = field(default_factory=list)

    @classmethod
    def from_json(cls, path: Path | None = None) -> "BookingStore":
        raw = json.loads((path or DATA_DIR / "schedule.json").read_text(encoding="utf-8"))
        today = date.fromisoformat(os.getenv("BOOKING_TODAY", raw["today"]))
        taken = {(b["date"], b["time"], b["master"]) for b in raw["existing_bookings"]}
        return cls(today, raw["opening_hour"], raw["last_start_hour"], raw["max_days_ahead"],
                   raw["services"], taken)

    def fresh_copy(self) -> "BookingStore":
        return copy.deepcopy(self)

    # ---- queries -------------------------------------------------------------------
    def bookable_dates(self) -> tuple[date, date]:
        return self.today + timedelta(days=1), self.today + timedelta(days=self.max_days_ahead)

    def hours(self) -> list[str]:
        return [f"{h:02d}:00" for h in range(self.opening_hour, self.last_start_hour + 1)]

    def free_slots(self, day: date, service_id: str, master: str | None = None) -> list[dict]:
        masters = self.services[service_id]["masters"]
        if master:
            masters = [m for m in masters if m == master]
        iso = day.isoformat()
        return [{"time": t, "master": m} for t in self.hours() for m in masters
                if (iso, t, m) not in self.taken]

    # ---- commands (only called after validation) -----------------------------------
    def book(self, *, service_id: str, day: date, time: str, master: str,
             client_name: str, phone: str) -> dict:
        key = (day.isoformat(), time, master)
        if key in self.taken:  # defensive: validation should have caught this
            raise ValueError("slot already taken")
        self.taken.add(key)
        booking = {
            "booking_id": f"B{len(self.bookings) + 1:04d}",
            "service_id": service_id,
            "service": self.services[service_id]["name"],
            "price_kzt": self.services[service_id]["price_kzt"],
            "date": day.isoformat(), "time": time, "master": master,
            "client_name": client_name, "phone": phone,
        }
        self.bookings.append(booking)
        return booking
