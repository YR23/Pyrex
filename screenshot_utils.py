"""Utilities for capturing screenshots from physical monitors."""

from __future__ import annotations

import json
import re
from pathlib import Path
import subprocess
from typing import Any, Literal, Union
import tkinter as tk

from PIL import Image, ImageEnhance, ImageOps, ImageTk


def screenshot_entire_monitor(output_path: str | Path, monitor_index: int = 2) -> Path:
    """Capture an entire physical monitor and save it to ``output_path``.

    This function is macOS-specific and uses the native ``screencapture`` command.
    By default it targets monitor ``2`` (commonly the first external display),
    while monitor ``1`` is usually the primary built-in display.

    Args:
        output_path: Destination image path (for example: ``"monitor.png"``).
        monitor_index: 1-based monitor index used by ``screencapture -D``.

    Returns:
        The resolved output path where the screenshot was saved.

    Raises:
        RuntimeError: If the screenshot command fails.
        ValueError: If ``monitor_index`` is less than 1.
    """
    if monitor_index < 1:
        raise ValueError("monitor_index must be >= 1")

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "screencapture",
        "-x",  # Disable shutter sound and UI.
        "-D",
        str(monitor_index),
        str(destination),
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        error_message = result.stderr.strip() or result.stdout.strip() or "Unknown error"
        raise RuntimeError(f"Failed to capture monitor {monitor_index}: {error_message}")

    return destination


def crop_monitor_screenshot(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    source_path: str | Path = "monitor_screenshot.png",
    output_path: str | Path = "cropped_monitor_screenshot.png",
    save_to_file: bool = True,
) -> Union[Path, Image.Image]:
    """Crop a rectangular area from a monitor screenshot.

    Args:
        x0: Left coordinate of the crop rectangle.
        y0: Top coordinate of the crop rectangle.
        x1: Right coordinate of the crop rectangle.
        y1: Bottom coordinate of the crop rectangle.
        source_path: Full screenshot file to crop from.
        output_path: Where to save the cropped image when ``save_to_file=True``.
        save_to_file: If True, save to disk and return output path.
            If False, return the cropped ``PIL.Image`` in memory.

    Returns:
        ``Path`` when ``save_to_file=True``, otherwise ``PIL.Image.Image``.

    Raises:
        FileNotFoundError: If ``source_path`` does not exist.
        ValueError: If coordinates are invalid or outside image bounds.
    """
    if x0 >= x1 or y0 >= y1:
        raise ValueError("Invalid crop rectangle: require x0 < x1 and y0 < y1")

    source = Path(source_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Screenshot file not found: {source}")

    with Image.open(source) as image:
        cropped = crop_from_loaded_image(image, x0, y0, x1, y1)
        if not save_to_file:
            return cropped.copy()

        destination = Path(output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(destination)
        return destination


def crop_from_loaded_image(
    image: Image.Image,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> Image.Image:
    """Crop ``(x0, y0, x1, y1)`` from an already-opened PIL image (no file I/O)."""
    if x0 >= x1 or y0 >= y1:
        raise ValueError("Invalid crop rectangle: require x0 < x1 and y0 < y1")

    width, height = image.size
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        raise ValueError(
            "Crop rectangle is out of bounds for the source image size "
            f"({width}x{height})"
        )

    return image.crop((x0, y0, x1, y1))


def is_player_name_active(
    name_image: Image.Image,
    *,
    luminance_peak_threshold: int = 200,
) -> bool:
    """Whether the name label looks like the bright-white \"active\" style.

    CoinPoker-style UI: acting player names reach true white (luminance 255);
    folded / waiting seats use dimmed text capped around 128. We use the
    brightest grayscale value in the crop as a simple discriminator.
    """
    gray = name_image.convert("L")
    hist = gray.histogram()
    peak = max(i for i, count in enumerate(hist) if count)
    return peak >= luminance_peak_threshold


def detect_dealer_marker(dealer_crop: Image.Image) -> bool:
    """Return True if a ``D`` dealer button is visible in the crop (OCR, letter only)."""
    try:
        import pytesseract
    except ImportError:
        return False

    def ocr_d(gray: Image.Image) -> bool:
        w, h = gray.size
        scale = max(64 / max(h, 1), 64 / max(w, 1), 2.0)
        if scale > 1.01:
            gray = gray.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                Image.Resampling.LANCZOS,
            )
        for psm in (10, 7):
            raw = pytesseract.image_to_string(
                gray,
                config=f"--psm {psm} -c tessedit_char_whitelist=D",
            ).strip()
            if "D" in raw.upper():
                return True
        return False

    gray = dealer_crop.convert("L")
    if ocr_d(gray):
        return True
    # Fallback: inverted contrast (some table themes differ)
    alt = ImageOps.invert(gray.copy())
    alt = ImageEnhance.Contrast(alt).enhance(1.8)
    return ocr_d(alt)


_ACTION_CHAR_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ-"


def _normalize_action_ocr(raw: str) -> str:
    """Keep only A–Z and hyphen (e.g. ``ALL-IN``); uppercase."""
    allowed = frozenset(_ACTION_CHAR_WHITELIST)
    return "".join(c for c in raw.upper() if c in allowed)


def recognize_table_action(action_crop: Image.Image) -> str:
    """Read action label: capital letters and ``-`` only (e.g. ``ALL-IN``); strip other chars."""

    try:
        import pytesseract
    except ImportError:
        return ""

    def ocr_variant(gray: Image.Image) -> str:
        w, h = gray.size
        scale = max(72 / max(h, 1), 100 / max(w, 1), 3.0)
        if scale > 1.01:
            gray = gray.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                Image.Resampling.LANCZOS,
            )
        best = ""
        for psm in (7, 8):
            raw = pytesseract.image_to_string(
                gray,
                config=f"--psm {psm} -c tessedit_char_whitelist={_ACTION_CHAR_WHITELIST}",
            )
            candidate = _normalize_action_ocr(raw)
            if len(candidate) > len(best):
                best = candidate
        if not best:
            for psm in (7, 8):
                raw = pytesseract.image_to_string(gray, config=f"--psm {psm}")
                candidate = _normalize_action_ocr(raw)
                if len(candidate) > len(best):
                    best = candidate
        return best

    gray = action_crop.convert("L")
    out = ocr_variant(gray)
    if out:
        return out
    alt = ImageOps.invert(gray.copy())
    alt = ImageEnhance.Contrast(alt).enhance(2.0)
    return ocr_variant(alt)


def pick_crop_coordinates(
    source_path: str | Path = "monitor_screenshot.png",
) -> tuple[int, int, int, int]:
    """Interactively pick crop coordinates by clicking two points on the image.

    Click once for top-left and once for bottom-right. If clicks are reversed,
    the function normalizes coordinates before returning.
    """
    source = Path(source_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Screenshot file not found: {source}")

    with Image.open(source) as image:
        image_copy = image.copy()

    root = tk.Tk()
    root.title("Pick Crop Area: click top-left, then bottom-right")

    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    max_w = int(screen_w * 0.9)
    max_h = int(screen_h * 0.9)

    img_w, img_h = image_copy.size
    scale = min(max_w / img_w, max_h / img_h, 1.0)
    display_w = int(img_w * scale)
    display_h = int(img_h * scale)

    display_image = image_copy.resize((display_w, display_h), Image.Resampling.LANCZOS)
    tk_image = ImageTk.PhotoImage(display_image)

    canvas = tk.Canvas(root, width=display_w, height=display_h, cursor="crosshair")
    canvas.pack()
    canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)

    clicks: list[tuple[int, int]] = []

    def on_click(event: tk.Event) -> None:
        x = int(event.x / scale)
        y = int(event.y / scale)
        clicks.append((x, y))

        # Visual marker for each click in the preview window.
        marker_size = 4
        canvas.create_oval(
            event.x - marker_size,
            event.y - marker_size,
            event.x + marker_size,
            event.y + marker_size,
            fill="red",
            outline="red",
        )

        if len(clicks) == 2:
            root.quit()

    canvas.bind("<Button-1>", on_click)
    root.mainloop()
    root.destroy()

    if len(clicks) < 2:
        raise RuntimeError("Coordinate selection cancelled before two clicks were made")

    (ax, ay), (bx, by) = clicks
    x0, x1 = sorted((ax, bx))
    y0, y1 = sorted((ay, by))
    return x0, y0, x1, y1


def _parse_rect(region: object, path_hint: str) -> tuple[int, int, int, int]:
    if not isinstance(region, dict):
        raise TypeError(f"{path_hint} must be an object with x0, y0, x1, y1")
    for coord in ("x0", "y0", "x1", "y1"):
        if coord not in region:
            raise KeyError(f"Missing '{path_hint}.{coord}'")
    return (
        int(region["x0"]),
        int(region["y0"]),
        int(region["x1"]),
        int(region["y1"]),
    )


def load_crop_regions_config(
    json_path: str | Path,
) -> tuple[str, dict[str, dict[str, Any]]]:
    """Load ``crop_regions.json`` with per-seat ``name``, ``stack``, optional ``dealer`` / ``action``.

    Top-level ``source_image`` (default ``monitor_screenshot.png``). Every other
    top-level object is a *seat* with required ``name`` and ``stack`` rects; optional
    ``dealer`` and ``action`` rects for dealer chip and action text (caps + ``-``).
    """
    path = Path(json_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    source = str(data.get("source_image", "monitor_screenshot.png"))

    seats: dict[str, dict[str, Any]] = {}
    for key, value in data.items():
        if key == "source_image":
            continue
        if not isinstance(value, dict):
            raise TypeError(f"Seat '{key}' must be an object with name and stack")
        if "name" not in value or "stack" not in value:
            raise KeyError(f"Seat '{key}' must include 'name' and 'stack'")
        seat: dict[str, Any] = {
            "name": _parse_rect(value["name"], f"{key}.name"),
            "stack": _parse_rect(value["stack"], f"{key}.stack"),
        }
        if "dealer" in value:
            seat["dealer"] = _parse_rect(value["dealer"], f"{key}.dealer")
        if "action" in value:
            seat["action"] = _parse_rect(value["action"], f"{key}.action")
        seats[key] = seat

    if not seats:
        raise ValueError(
            f"No seat regions in {path}; add at least one seat (e.g. top_left) with name and stack."
        )

    return source, seats


def _prepare_stack_for_ocr_text(gray: Image.Image) -> Image.Image:
    """Upscale and normalize bright-on-dark digits (e.g. yellow stacks) for Tesseract."""
    w, h = gray.size
    scale = max(360 / max(w, 1), 120 / max(h, 1), 6.0)
    gray = gray.resize(
        (max(1, int(w * scale)), max(1, int(h * scale))),
        Image.Resampling.LANCZOS,
    )
    # Bright glyphs on dark felt → dark glyphs on light background (Tesseract prefers this).
    gray = ImageOps.invert(gray)
    gray = ImageOps.autocontrast(gray, cutoff=0)
    gray = ImageEnhance.Contrast(gray).enhance(2.2)
    pad = max(8, gray.height // 4)
    new = Image.new("L", (gray.width + 2 * pad, gray.height + 2 * pad), color=255)
    new.paste(gray, (pad, pad))
    return new


def _normalize_stack_string(value: str) -> str:
    return value.strip().replace(",", ".")


def _stack_ocr_score(value: str) -> tuple[int, int, int]:
    """Higher is better: prefer plausible chip strings with a decimal and more digits."""
    s = _normalize_stack_string(value)
    if not s or not re.fullmatch(r"[\d.]+", s):
        return (-1, 0, 0)
    digit_count = sum(1 for c in s if c.isdigit())
    has_dot = 1 if "." in s else 0
    # Penalize lone "." or ".." junk
    if s == "." or s.count(".") > 1:
        return (-1, 0, 0)
    return (has_dot, digit_count, len(s))


def _merge_stack_ocr_candidates(enhanced: str, simple: str) -> str:
    """Pick the better of enhanced vs simple read (they disagree on some crops)."""
    a, b = _normalize_stack_string(enhanced), _normalize_stack_string(simple)
    if not a:
        return b
    if not b:
        return a
    score_a, score_b = _stack_ocr_score(a), _stack_ocr_score(b)
    if score_a > score_b:
        return a
    if score_b > score_a:
        return b
    # Tie: prefer simple — enhanced prep can be empty or wrong on some crops (e.g. lone digit).
    return b


def _ocr_stack_last_resort(image: Image.Image) -> str:
    """When merged stack OCR is empty, try heavy upscale + alternate page modes."""
    try:
        import pytesseract
    except ImportError:
        return ""

    gray = image.convert("L")
    w, h = gray.size
    scale = max(4.0, 100 / max(h, 1), 320 / max(w, 1))
    big = gray.resize(
        (max(1, int(w * scale)), max(1, int(h * scale))),
        Image.Resampling.LANCZOS,
    )
    for cfg in (
        "--psm 10 -c tessedit_char_whitelist=0123456789.,",
        "--psm 7 -c tessedit_char_whitelist=0123456789.,",
        "--psm 10",
    ):
        raw = pytesseract.image_to_string(big, config=cfg).strip()
        s = _normalize_stack_string(raw)
        if s and re.fullmatch(r"[\d.]+", s) and s != ".":
            return s
    return ""


def recognize_text_from_image(
    image: Image.Image,
    *,
    kind: Literal["name", "stack"] = "name",
    stack: Literal["text", "simple"] = "text",
) -> str:
    """Run OCR on a PIL image (requires ``pytesseract`` and system ``tesseract``).

    For ``kind="stack"``, ``stack="text"`` (default) runs the enhanced pipeline and
    merges it with the simple read when the simple line keeps a clearer decimal
    (e.g. ``0.6``). Use ``stack="simple"`` for the previous light-touch pipeline only.
    """
    try:
        import pytesseract
    except ImportError as exc:
        raise ImportError(
            "Install pytesseract (`pip install pytesseract`) and the Tesseract "
            "binary (e.g. `brew install tesseract`)."
        ) from exc

    gray = image.convert("L")
    w, h = gray.size

    if kind == "name":
        if w < 120 or h < 24:
            scale = max(120 / w, 24 / h, 2.0)
            gray = gray.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                Image.Resampling.LANCZOS,
            )
        config = "--psm 7"
        return pytesseract.image_to_string(gray, config=config).strip()

    # kind == "stack"
    stack_config_simple = "--psm 7 -c tessedit_char_whitelist=0123456789.,"
    stack_config_text = "--psm 7 -c tessedit_char_whitelist=0123456789.,"

    if stack == "text":
        ocr_input = _prepare_stack_for_ocr_text(gray)
        text_out = pytesseract.image_to_string(ocr_input, config=stack_config_text).strip()

        gray_simple = image.convert("L")
        ws, hs = gray_simple.size
        # Short stack strips (e.g. hero chip count) need upscale even when width >= 120.
        if ws < 120 or hs < 24 or hs < 72:
            scale = max(120 / max(ws, 1), 24 / max(hs, 1), 72 / max(hs, 1), 2.0)
            gray_simple = gray_simple.resize(
                (max(1, int(ws * scale)), max(1, int(hs * scale))),
                Image.Resampling.LANCZOS,
            )
        simple_out = pytesseract.image_to_string(
            gray_simple, config=stack_config_simple
        ).strip()

        merged = _merge_stack_ocr_candidates(text_out, simple_out)
        norm = _normalize_stack_string(merged)
        if norm and _stack_ocr_score(norm)[0] >= 0:
            return norm
        return _ocr_stack_last_resort(image)

    if w < 120 or h < 24:
        scale = max(120 / w, 24 / h, 2.0)
        gray = gray.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.Resampling.LANCZOS,
        )
    return pytesseract.image_to_string(gray, config=stack_config_simple).strip()
