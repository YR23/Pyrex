"""Table seat model after OCR."""

from __future__ import annotations

from dataclasses import dataclass

# Physical order around the table, clockwise (12 o'clock → … →). Adjust if your layout differs.
SEAT_ORDER_CLOCKWISE = (
    "top_middle",
    "top_right",
    "bottom_right",
    "bottom_middle",
    "bottom_left",
    "top_left",
)

# Six-max labels starting at dealer (button), then clockwise.
SIX_MAX_POSITIONS = ("BTN", "SB", "BB", "UTG", "MP", "CO")


@dataclass
class Player:
    """One player seat: name, stack, active styling, dealer chip, and table position."""

    seat: str
    name: str
    stack: str
    active: bool
    dealer: bool = False
    position: str = "?"
    action: str = ""
    card_left: str = ""
    card_right: str = ""
    hole_cards: str = ""

    @property
    def status(self) -> str:
        return "Active" if self.active else "Inactive"

    @property
    def last_action(self) -> str:
        if not self.active:
            return "FOLD"
        return ""


def assign_six_max_positions(players: list[Player]) -> None:
    """Set ``position`` from ``dealer`` and ``SEAT_ORDER_CLOCKWISE`` (6-max).

    Dealer seat is BTN; next clockwise seats are SB, BB, UTG, MP, CO.
    If no dealer is detected, every ``position`` stays ``?``.
    If multiple seats report ``dealer``, the first clockwise from ``top_middle`` wins.
    """
    n = len(SIX_MAX_POSITIONS)
    by_seat = {p.seat: p for p in players}
    dealer_indices = [
        i
        for i, seat in enumerate(SEAT_ORDER_CLOCKWISE)
        if seat in by_seat and by_seat[seat].dealer
    ]
    if not dealer_indices:
        for p in players:
            p.position = "?"
        return

    btn_index = dealer_indices[0]

    for p in players:
        if p.seat not in SEAT_ORDER_CLOCKWISE:
            p.position = "?"
            continue
        seat_index = SEAT_ORDER_CLOCKWISE.index(p.seat)
        offset = (seat_index - btn_index) % n
        p.position = SIX_MAX_POSITIONS[offset]
