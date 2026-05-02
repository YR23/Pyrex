"""Shared constants for table / UI parsing."""

from __future__ import annotations

# You always sit here; hole-card crops in JSON apply only to this seat.
HERO_SEAT = "bottom_middle"
HERO_HAND_KEYS: tuple[str, ...] = ("left_hand", "right_hand", "hero_hand")

# Single-card rank labels (order: longer numerals first so ``10`` beats ``1``).
CANONICAL_RANKS: tuple[str, ...] = (
    "10",
    "9",
    "8",
    "7",
    "6",
    "5",
    "4",
    "3",
    "2",
    "1",
    "A",
    "K",
    "Q",
    "J",
)

# Canonical action labels (CoinPoker-style). Used to normalize OCR noise.
CANONICAL_ACTIONS: tuple[str, ...] = (
    "CALL",
    "FOLD",
    "BET",
    "RAISE",
    "SB",
    "BB",
    "DISCONNECT",
    "MUCK",
    "ALL-IN",
)


def _compact_alnum(s: str) -> str:
    """Uppercase and strip hyphens for loose ``ALL-IN`` / ``ALLIN`` matching."""
    return s.replace("-", "").upper()


def resolve_table_action(ocr: str) -> str:
    """Map raw OCR text to a canonical action when possible.

    If some ``CANONICAL_ACTIONS`` entry appears as a **contiguous substring** of
    ``ocr`` (after strip / upper), return the **longest** such match (e.g.
    ``CALLO`` → ``CALL``).

    Also matches when hyphens differ (e.g. ``ALLIN`` → ``ALL-IN``) by comparing
    compact forms for that substring rule.

    If no canonical substring is found, return ``ocr`` unchanged (e.g. ``CAL``).
    """
    s = ocr.strip().upper()
    if not s:
        return ""

    compact_s = _compact_alnum(s)
    matches: list[str] = []

    for canonical in CANONICAL_ACTIONS:
        if canonical in s:
            matches.append(canonical)
            continue
        cc = _compact_alnum(canonical)
        if cc and cc in compact_s:
            matches.append(canonical)

    if not matches:
        return s

    return max(matches, key=len)


def _normalize_rank_token(raw: str) -> str:
    """Keep rank-like characters only (digits + JQKA)."""
    return "".join(c.upper() for c in raw if c.upper() in "0123456789JQKA")


def _truncate_digit_run_if_gt_20(s: str) -> str:
    """If ``s`` is all digits and value > 20, keep first digit only (OCR glitches like ``30``)."""
    if s.isdigit() and int(s) > 20:
        return s[0]
    return s


def normalize_hole_card_ranks(ocr: str) -> str:
    """Strip non-rank characters (for a combined ``hero_hand`` crop, no single-rank pick)."""
    return _truncate_digit_run_if_gt_20(_normalize_rank_token(ocr.strip()))


def resolve_hole_card_rank(ocr: str) -> str:
    """Map one-card OCR to a canonical rank from ``CANONICAL_RANKS`` when possible.

    If the normalized token is **all digits** and its integer value is **greater than
    20**, only the **first digit** is kept (e.g. ``30`` → ``3``) before substring matching.

    Same rule as actions: if some rank string is a **contiguous substring** of that
    string, return the **longest** match (e.g. ``1O`` after ``O→0`` → ``10``).

    If none match, return the normalized OCR (e.g. partial ``XY``).
    """
    fixed = ocr.strip().upper().replace("O", "0")
    s = _normalize_rank_token(fixed)
    if not s:
        return ""

    s = _truncate_digit_run_if_gt_20(s)

    matches = [r for r in CANONICAL_RANKS if r in s]
    if not matches:
        return s

    return max(matches, key=len)
