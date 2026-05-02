"""Per-hand / per-street session tracking (preflop-only action counts for now)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

from consts import CANONICAL_ACTIONS, HERO_SEAT
from player import Player, format_hole_xy

# First voluntary preflop actor is LJ; then clockwise to BB.
PREFLOP_ACTING_ORDER: tuple[str, ...] = ("LJ", "HJ", "CO", "BTN", "SB", "BB")

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
    """Per-seat counters for the **current hero deal** only (reset when hero hole cards change)."""

    seat: str
    hands_played: int = 1
    action_counts: dict[str, int] = field(default_factory=dict)


def _hero_hole_signature(hero: Player) -> str:
    """Stable key for hero cards; empty if either side is missing."""
    left, right = hero.card_left.strip(), hero.card_right.strip()
    if not left or not right:
        return ""
    return f"{left}|{right}"


def _effective_visible_action(p: Player) -> str:
    """Best label for what this seat has done this street (OCR + inactive ⇒ fold)."""
    if not p.active:
        return "FOLD"
    a = (p.action or "").strip()
    return a if a else "—"


def build_table_snapshot(
    players: list[Player],
    *,
    situation: str | None,
    spot: str | None = None,
    last_aggressor: str | None = None,
) -> dict[str, dict[str, object]]:
    """Compact per-seat snapshot for strategy decisions only."""
    out: dict[str, dict[str, object]] = {}
    for p in players:
        action_show = p.action if p.action else "—"
        row: dict[str, object] = {
            "position": p.position,
            "action": action_show,
        }
        if p.seat == HERO_SEAT:
            row["my_turn"] = p.my_turn
            row["situation"] = (situation or "") if p.my_turn else ""
            row["spot"] = (spot or "") if p.my_turn else ""
            row["last_aggressor"] = (last_aggressor or "") if p.my_turn else ""
        out[p.seat] = row
    return out


def snapshot_fingerprint(snapshot: dict[str, dict[str, object]]) -> str:
    return json.dumps(snapshot, sort_keys=True, ensure_ascii=False)


def preflop_situation_before_hero(players: list[Player]) -> str:
    """Describe preflop action at every seat that acts before hero (requires ``assign_six_max_positions``)."""
    hero = next((p for p in players if p.seat == HERO_SEAT), None)
    if hero is None:
        return "Hero seat not in snapshot."
    if hero.position not in PREFLOP_ACTING_ORDER:
        return (
            "Cannot describe preflop order before you: hero position is unknown "
            "(dealer chip not detected or ambiguous)."
        )
    idx = PREFLOP_ACTING_ORDER.index(hero.position)
    prior = PREFLOP_ACTING_ORDER[:idx]
    if not prior:
        return (
            "Preflop: you are first to act (LJ). Only blinds are posted before you; "
            "no one has voluntarily acted yet."
        )
    by_position = {p.position: p for p in players}
    segments: list[str] = []
    for pos in prior:
        p = by_position.get(pos)
        if p is None:
            continue
        segments.append(f"{pos} {_effective_visible_action(p)}")
    chain = " → ".join(segments) if segments else "(no seats)"
    return f"Preflop, action before you: {chain}."


def preflop_spot_for_hero(players: list[Player]) -> str:
    """Spot label at hero decision point: RFI / VS-RFI / VS-3-BET / VS-4-BET / VS-RAISE-CALL."""
    hero = next((p for p in players if p.seat == HERO_SEAT), None)
    if hero is None or not hero.my_turn:
        return ""
    if hero.position not in PREFLOP_ACTING_ORDER:
        return "UNKNOWN"

    idx = PREFLOP_ACTING_ORDER.index(hero.position)
    prior_positions = PREFLOP_ACTING_ORDER[:idx]
    by_position = {p.position: p for p in players}

    raises = 0
    calls = 0
    for pos in prior_positions:
        p = by_position.get(pos)
        if p is None:
            continue
        eff = _effective_visible_action(p)
        label = _canonical_action_label(eff)
        if label in ("RAISE", "BET", "ALL-IN"):
            raises += 1
        elif label == "CALL":
            calls += 1

    if raises == 0:
        return "RFI"
    if raises == 1 and calls > 0:
        return "VS-RAISE-CALL"
    if raises == 1:
        return "VS-RFI"
    if raises == 2:
        return "VS-3-BET"
    return "VS-4-BET"


def preflop_last_aggressor_before_hero(players: list[Player]) -> str:
    """Return position of last raise-like action before hero, else ``""``."""
    hero = next((p for p in players if p.seat == HERO_SEAT), None)
    if hero is None or hero.position not in PREFLOP_ACTING_ORDER:
        return ""
    idx = PREFLOP_ACTING_ORDER.index(hero.position)
    prior_positions = PREFLOP_ACTING_ORDER[:idx]
    by_position = {p.position: p for p in players}

    last: str = ""
    for pos in prior_positions:
        p = by_position.get(pos)
        if p is None:
            continue
        eff = _effective_visible_action(p)
        label = _canonical_action_label(eff)
        if label in ("RAISE", "BET", "ALL-IN"):
            last = pos
    return last


def _canonical_action_label(action: str) -> str | None:
    """Return the canonical action string, or None if OCR is not a known action."""
    s = action.strip()
    if not s:
        return None
    return _CANONICAL_BY_UPPER.get(s.upper())


class TableSessionTracker:
    """Tracks the **current hand** until hero acts: one persisted session file, reset on new hero cards."""

    def __init__(self) -> None:
        self.current_hand = Hand()
        self._last_hero_sig: str | None = None
        self._counted_actions: dict[str, set[str]] = {}
        self.stats: dict[str, PlayerHandStats] = {}
        self.table_snapshot: dict[str, dict[str, object]] = {}
        self.table_snapshot_fingerprint: str | None = None

    def _ensure_stats(self, seat: str) -> None:
        if seat not in self.stats:
            self.stats[seat] = PlayerHandStats(seat=seat)

    def _reset_counted(self, seats: tuple[str, ...]) -> None:
        self._counted_actions = {seat: set() for seat in seats}

    def _on_new_hand(self, seats: tuple[str, ...], sig: str) -> None:
        """New hero hole cards: clear prior deal; file should describe only this hand until hero's turn."""
        self._reset_counted(seats)
        self.stats = {
            seat: PlayerHandStats(seat=seat, hands_played=1) for seat in seats
        }
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

    def saved_hero_my_turn(self) -> bool:
        """From last persisted ``table_snapshot`` (used to skip live capture while acting)."""
        hero = self.table_snapshot.get(HERO_SEAT) if self.table_snapshot else None
        if not hero:
            return False
        return bool(hero.get("my_turn"))

    def saved_hero_seat_active(self) -> bool:
        """Whether last saved snapshot had hero as ``Active`` (``Inactive`` ≈ folded)."""
        hero = self.table_snapshot.get(HERO_SEAT) if self.table_snapshot else None
        if not hero:
            return True
        return str(hero.get("status", "")).strip().lower() == "active"

    def set_table_snapshot_if_changed(
        self, snapshot: dict[str, dict[str, object]]
    ) -> bool:
        """Update in-memory snapshot only when content differs. Returns whether it changed."""
        fp = snapshot_fingerprint(snapshot)
        if fp == self.table_snapshot_fingerprint:
            return False
        self.table_snapshot = snapshot
        self.table_snapshot_fingerprint = fp
        return True

    def report_lines(self) -> list[str]:
        """Human-readable summary lines for logging."""
        sig = self.current_hand.hero_hole_signature
        xy = ""
        if sig and "|" in sig:
            left, right = sig.split("|", 1)
            xy = format_hole_xy(left, right)
        lines = [
            "--- Current hand session (preflop; resets when hero hole cards change) ---",
            f"Current street: {self.current_hand.street.name}",
            f"Hero hole signature: {sig or '—'}",
        ]
        if xy:
            lines.append(f"Hero hole (XYz): {xy}")
        for seat in sorted(self.stats.keys()):
            s = self.stats[seat]
            parts = [f"deal_session={s.hands_played}"]
            if s.action_counts:
                ac = ", ".join(f"{k}={v}" for k, v in sorted(s.action_counts.items()))
                parts.append(f"actions: {ac}")
            lines.append(f"  {seat}: " + " | ".join(parts))
        return lines

    def to_json_dict(self) -> dict:
        pos_by_seat: dict[str, str] = {}
        for seat, payload in self.table_snapshot.items():
            if not isinstance(payload, dict):
                continue
            pos = str(payload.get("position", "")).strip()
            if pos:
                pos_by_seat[seat] = pos
        counted_by_position: dict[str, set[str]] = {}
        for seat, labels in self._counted_actions.items():
            key = pos_by_seat.get(seat, seat)
            bucket = counted_by_position.setdefault(key, set())
            bucket.update(labels)

        return {
            "version": 2,
            "last_hero_sig": self._last_hero_sig,
            "current_hand": {
                "street": self.current_hand.street.name,
                "hero_hole_signature": self.current_hand.hero_hole_signature,
            },
            "counted_actions": {
                pos: sorted(labels) for pos, labels in counted_by_position.items()
            },
            "stats": {
                seat: {
                    "seat": s.seat,
                    "hands_played": s.hands_played,
                    "action_counts": dict(s.action_counts),
                }
                for seat, s in self.stats.items()
            },
            "table_snapshot": self.table_snapshot,
            "table_snapshot_fingerprint": self.table_snapshot_fingerprint,
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
        raw_snap = data.get("table_snapshot") or {}
        if isinstance(raw_snap, dict):
            t.table_snapshot = {
                str(seat): dict(payload) if isinstance(payload, dict) else {}
                for seat, payload in raw_snap.items()
            }
        t.table_snapshot_fingerprint = data.get("table_snapshot_fingerprint")
        if isinstance(t.table_snapshot_fingerprint, str):
            t.table_snapshot_fingerprint = t.table_snapshot_fingerprint.strip() or None
        else:
            t.table_snapshot_fingerprint = None
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
        pos_to_seat: dict[str, str] = {}
        for seat, payload in t.table_snapshot.items():
            if not isinstance(payload, dict):
                continue
            pos = str(payload.get("position", "")).strip()
            if pos:
                pos_to_seat[pos] = seat
        counted: dict[str, set[str]] = {}
        for key, labels in raw_counted.items():
            seat_key = pos_to_seat.get(str(key), str(key))
            counted[seat_key] = set(labels)
        t._counted_actions = counted
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
