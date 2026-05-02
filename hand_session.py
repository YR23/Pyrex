"""Per-hand / per-street session tracking (preflop-only action counts for now)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

from consts import CANONICAL_ACTIONS, HERO_SEAT
from player import Player, format_hole_xy

DEFAULT_SESSION_PATH = Path(__file__).resolve().parent / "hand_session.json"

_CANONICAL_BY_UPPER: dict[str, str] = {c.upper(): c for c in CANONICAL_ACTIONS}
# Blinds (SB/BB) are posted labels, not voluntary actions — omit from session stats.
_COUNTABLE = frozenset(c for c in CANONICAL_ACTIONS if c not in ("SB", "BB"))


class Street(Enum):
    """Betting round order; only ``PREFLOP`` is used until board crops exist."""

    PREFLOP = auto()
    FLOP = auto()
    TURN = auto()
    RIVER = auto()


@dataclass
class Hand:
    """One deal from hero hole cards through river (streets advance later)."""

    street: Street = Street.PREFLOP
    hero_hole_signature: str = ""


@dataclass
class PlayerHandStats:
    """Session counters for a single seat."""

    seat: str
    hands_played: int = 0
    action_counts: dict[str, int] = field(default_factory=dict)


def _hero_hole_signature(hero: Player) -> str:
    """Stable key for hero cards; empty if either side is missing."""
    left, right = hero.card_left.strip(), hero.card_right.strip()
    if not left or not right:
        return ""
    return f"{left}|{right}"


def _canonical_action_label(action: str) -> str | None:
    """Return the canonical action string, or None if OCR is not a known action."""
    s = action.strip()
    if not s:
        return None
    return _CANONICAL_BY_UPPER.get(s.upper())


class TableSessionTracker:
    """Tracks new hands (hero hole cards change) and one count per action per hand."""

    def __init__(self) -> None:
        self.current_hand = Hand()
        self._last_hero_sig: str | None = None
        self._counted_actions: dict[str, set[str]] = {}
        self.stats: dict[str, PlayerHandStats] = {}

    def _ensure_stats(self, seat: str) -> None:
        if seat not in self.stats:
            self.stats[seat] = PlayerHandStats(seat=seat)

    def _reset_counted(self, seats: tuple[str, ...]) -> None:
        self._counted_actions = {seat: set() for seat in seats}

    def _on_new_hand(self, seats: tuple[str, ...], sig: str) -> None:
        for seat in seats:
            self._ensure_stats(seat)
            self.stats[seat].hands_played += 1
        self._reset_counted(seats)
        self.current_hand = Hand(street=Street.PREFLOP, hero_hole_signature=sig)

    def _count_preflop_actions(self, players: list[Player]) -> None:
        for p in players:
            label = _canonical_action_label(p.action)
            if label is None or label not in _COUNTABLE:
                continue
            counted = self._counted_actions.setdefault(p.seat, set())
            if label in counted:
                continue
            self._ensure_stats(p.seat)
            stats = self.stats[p.seat]
            stats.action_counts[label] = stats.action_counts.get(label, 0) + 1
            counted.add(label)

    def ingest(self, players: list[Player]) -> None:
        """Call once per table snapshot after OCR fills ``Player`` fields."""
        seats = tuple(p.seat for p in players)
        for seat in seats:
            self._ensure_stats(seat)

        hero = next((p for p in players if p.seat == HERO_SEAT), None)
        sig = _hero_hole_signature(hero) if hero else ""

        if sig:
            if self._last_hero_sig != sig:
                self._on_new_hand(seats, sig)
            self._last_hero_sig = sig
            self.current_hand.hero_hole_signature = sig

        self._count_preflop_actions(players)

    def report_lines(self) -> list[str]:
        """Human-readable summary lines for logging."""
        sig = self.current_hand.hero_hole_signature
        xy = ""
        if sig and "|" in sig:
            left, right = sig.split("|", 1)
            xy = format_hole_xy(left, right)
        lines = [
            "--- Hand session (preflop; new hand = hero hole cards changed) ---",
            f"Current street: {self.current_hand.street.name}",
            f"Hero hole signature: {sig or '—'}",
        ]
        if xy:
            lines.append(f"Hero hole (XYz): {xy}")
        for seat in sorted(self.stats.keys()):
            s = self.stats[seat]
            parts = [f"hands_played={s.hands_played}"]
            if s.action_counts:
                ac = ", ".join(f"{k}={v}" for k, v in sorted(s.action_counts.items()))
                parts.append(f"actions: {ac}")
            lines.append(f"  {seat}: " + " | ".join(parts))
        return lines

    def to_json_dict(self) -> dict:
        return {
            "version": 1,
            "last_hero_sig": self._last_hero_sig,
            "current_hand": {
                "street": self.current_hand.street.name,
                "hero_hole_signature": self.current_hand.hero_hole_signature,
            },
            "counted_actions": {
                seat: sorted(labels) for seat, labels in self._counted_actions.items()
            },
            "stats": {
                seat: {
                    "seat": s.seat,
                    "hands_played": s.hands_played,
                    "action_counts": dict(s.action_counts),
                }
                for seat, s in self.stats.items()
            },
        }

    def save(self, path: str | Path = DEFAULT_SESSION_PATH) -> Path:
        """Persist session state for the next run."""
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_json_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return destination

    @classmethod
    def load(cls, path: str | Path = DEFAULT_SESSION_PATH) -> TableSessionTracker:
        """Restore session from disk, or start empty if the file is missing."""
        source = Path(path).expanduser().resolve()
        if not source.exists():
            return cls()
        data = json.loads(source.read_text(encoding="utf-8"))
        t = cls()
        t._last_hero_sig = data.get("last_hero_sig")
        ch = data.get("current_hand") or {}
        street_name = ch.get("street", Street.PREFLOP.name)
        try:
            street = Street[street_name]
        except KeyError:
            street = Street.PREFLOP
        t.current_hand = Hand(
            street=street,
            hero_hole_signature=str(ch.get("hero_hole_signature", "")),
        )
        raw_counted = data.get("counted_actions") or {}
        t._counted_actions = {seat: set(labels) for seat, labels in raw_counted.items()}
        raw_stats = data.get("stats") or {}
        for seat, payload in raw_stats.items():
            if not isinstance(payload, dict):
                continue
            t.stats[seat] = PlayerHandStats(
                seat=str(payload.get("seat", seat)),
                hands_played=int(payload.get("hands_played", 0)),
                action_counts=dict(payload.get("action_counts") or {}),
            )
        return t
