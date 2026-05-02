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


def _split_rank_color(card: str) -> tuple[str, str]:
    """Parse ``{rank}-{color}`` from a single hole crop label."""
    s = card.strip()
    if not s:
        return "", ""
    if "-" not in s:
        return s, ""
    rank, color = s.rsplit("-", 1)
    return rank.strip(), color.strip()


def _rank_for_xy_notation(rank: str) -> str:
    """Shorthand rank for ``XYz`` (ten → ``T``)."""
    if rank == "10":
        return "T"
    return rank


def format_hole_xy(card_left: str, card_right: str) -> str:
    """Hole cards as ``XYz``: left rank ``X``, right rank ``Y``, suffix ``z``.

    ``z`` is ``s`` if both non-pair cards share the same dominant color (suited),
    ``o`` if ranks differ and colors differ or color is unknown, and ``""`` for a
    pocket pair (same rank).

    Returns ``""`` if either side is missing.
    """
    lr, lc = _split_rank_color(card_left)
    rr, rc = _split_rank_color(card_right)
    if not lr or not rr:
        return ""
    xl = _rank_for_xy_notation(lr)
    yr = _rank_for_xy_notation(rr)
    if lr == rr:
        return f"{xl}{yr}"
    if lc and rc and lc == rc:
        return f"{xl}{yr}s"
    return f"{xl}{yr}o"


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
    my_turn: bool = False

    @property
    def status(self) -> str:
        return "Active" if self.active else "Inactive"

    @property
    def last_action(self) -> str:
        if not self.active:
            return "FOLD"
        return ""

    @property
    def hole_xy(self) -> str:
        """``XYz`` shorthand from ``card_left`` / ``card_right`` (see ``format_hole_xy``)."""
        return format_hole_xy(self.card_left, self.card_right)


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
