"""Stable fingerprint for street UI crops (preflop vs later) without storing reference PNGs.

Raw SHA-256 of encoded PNG bytes changes with compression and tiny pixel drift. We use an
8×8 ``average hash`` (64 bits): resize to grayscale 8×8, each bit is ``pixel >= mean``.
Compare runs to a reference with **Hamming distance** (tune ``street_preflop_hamming_max``).
"""

from __future__ import annotations

from PIL import Image


def street_crop_average_hash_u64(image: Image.Image) -> int:
    """64-bit average hash (aHash); identical pipeline ⇒ comparable across captures."""
    small = image.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
    pixels = small.getdata()
    mean = sum(pixels) / 64.0
    h = 0
    for i, p in enumerate(pixels):
        if p >= mean:
            h |= 1 << i
    return h & 0xFFFFFFFFFFFFFFFF


def format_u64_hash(h: int) -> str:
    return f"{h & 0xFFFFFFFFFFFFFFFF:016x}"


def parse_u64_hash_hex(s: str) -> int:
    t = s.strip().lower()
    if len(t) != 16:
        raise ValueError("street_preflop_ahash must be 16 hex characters (64-bit)")
    return int(t, 16) & 0xFFFFFFFFFFFFFFFF


def hamming_u64(a: int, b: int) -> int:
    return ((a ^ b) & 0xFFFFFFFFFFFFFFFF).bit_count()


def is_preflop_street_crop(
    image: Image.Image,
    *,
    ref_hex: str | None,
    hamming_max: int,
) -> tuple[bool | None, int, int]:
    """Return ``(is_preflop, hamming_distance, current_hash_u64)``.

    ``is_preflop`` is ``None`` when no ``ref_hex`` is configured.
    """
    cur = street_crop_average_hash_u64(image)
    if not ref_hex or not str(ref_hex).strip():
        return None, 0, cur
    ref = parse_u64_hash_hex(str(ref_hex))
    dist = hamming_u64(cur, ref)
    return dist <= hamming_max, dist, cur
