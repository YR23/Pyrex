"""Read crop regions from JSON, OCR — or interactively pick coordinates on the image."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from screenshot_utils import (
    crop_from_loaded_image,
    load_crop_regions_config,
    pick_crop_coordinates,
    recognize_text_from_image,
)


def _pick_and_print(source_path: Path, label: str) -> tuple[int, int, int, int]:
    print(f"{label}: click top-left, then bottom-right on the image window.")
    x0, y0, x1, y1 = pick_crop_coordinates(source_path)
    print(f"  {label}: x0={x0}, y0={y0}, x1={x1}, y1={y1}")
    print(
        f'  JSON: "{label}": {{"x0": {x0}, "y0": {y0}, "x1": {x1}, "y1": {y1}}},'
    )
    return x0, y0, x1, y1


def _process_one_seat(
    base: Image.Image,
    seat_label: str,
    regions: dict[str, tuple[int, int, int, int]],
    debug_dir: Path,
) -> tuple[str, str, str, Path, Path]:
    """Crop one seat, OCR name/stack, save debug PNGs (safe to run in parallel threads)."""
    nx0, ny0, nx1, ny1 = regions["name"]
    sx0, sy0, sx1, sy1 = regions["stack"]

    name_image = crop_from_loaded_image(base, nx0, ny0, nx1, ny1)
    stack_image = crop_from_loaded_image(base, sx0, sy0, sx1, sy1)

    player_name = recognize_text_from_image(name_image, kind="name")
    stack = recognize_text_from_image(stack_image, kind="stack")

    name_debug_path = debug_dir / f"{seat_label}_name.png"
    stack_debug_path = debug_dir / f"{seat_label}_stack.png"
    name_image.save(name_debug_path)
    stack_image.save(stack_debug_path)

    return seat_label, player_name, stack, name_debug_path, stack_debug_path


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR hero name/stack from crops, or pick coords on screenshot.")
    parser.add_argument(
        "--pick",
        action="store_true",
        help="Open the screenshot once; pick one rectangle; print coords for crop_regions.json.",
    )
    parser.add_argument(
        "--pick-both",
        action="store_true",
        help="Pick name rectangle, then stack rectangle; print JSON for a seat (e.g. top_left).",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="monitor_screenshot.png",
        help="Screenshot file to use with --pick / --pick-both (relative to this folder).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    source_path = (repo_root / args.image).resolve()

    if args.pick_both:
        nx0, ny0, nx1, ny1 = _pick_and_print(source_path, "name")
        sx0, sy0, sx1, sy1 = _pick_and_print(source_path, "stack")
        print(
            'Paste under a seat key, e.g. "top_left": '
            f'{{ "name": {{"x0": {nx0}, "y0": {ny0}, "x1": {nx1}, "y1": {ny1}}}, '
            f'"stack": {{"x0": {sx0}, "y0": {sy0}, "x1": {sx1}, "y1": {sy1}}} }},'
        )
        return

    if args.pick:
        _pick_and_print(source_path, "region")
        print('Paste under a seat as "name" or "stack", e.g. top_left.name.')
        return

    config_path = repo_root / "crop_regions.json"
    source_rel, seats = load_crop_regions_config(config_path)
    image_path = repo_root / source_rel

    with Image.open(image_path) as monitor_image:
        base = monitor_image.copy()

    debug_dir = repo_root / "debug_crops"
    debug_dir.mkdir(parents=True, exist_ok=True)

    seat_items = list(seats.items())
    max_workers = min(8, len(seat_items)) if seat_items else 1

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(_process_one_seat, base, seat_label, regions, debug_dir)
            for seat_label, regions in seat_items
        ]
        for fut in futures:
            seat_label, player_name, stack, name_debug_path, stack_debug_path = fut.result()
            print(f"[{seat_label}] Player: {player_name}")
            print(f"[{seat_label}] Stack: {stack}")
            print(f"[{seat_label}] Debug crops: {name_debug_path} , {stack_debug_path}")


if __name__ == "__main__":
    main()
