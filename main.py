"""Read crop regions from JSON, OCR — or interactively pick coordinates on the image."""

import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from consts import (
    HERO_HAND_OCR_ORDER,
    HERO_SEAT,
    normalize_hole_card_ranks,
    resolve_hole_card_rank,
    resolve_hole_card_rank_at_least,
    resolve_table_action,
)
from hand_session import DEFAULT_SESSION_PATH, TableSessionTracker
from player import Player, assign_six_max_positions
from screenshot_utils import (
    crop_from_loaded_image,
    detect_dealer_marker,
    detect_hero_my_turn_from_strip,
    dominant_hole_card_suit_color,
    is_player_name_active,
    load_crop_regions_config,
    pick_crop_coordinates,
    recognize_hole_card_crop,
    recognize_table_action,
    recognize_text_from_image,
    screenshot_entire_monitor,
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
    regions: dict,
    crop_lock: threading.Lock,
) -> Player:
    """Crop one seat in memory, OCR name/stack, optional dealer marker, and hero extras."""

    def safe_crop(x0: int, y0: int, x1: int, y1: int) -> Image.Image:
        with crop_lock:
            return crop_from_loaded_image(base, x0, y0, x1, y1)

    nx0, ny0, nx1, ny1 = regions["name"]
    sx0, sy0, sx1, sy1 = regions["stack"]

    name_image = safe_crop(nx0, ny0, nx1, ny1)
    stack_image = safe_crop(sx0, sy0, sx1, sy1)

    player_name = recognize_text_from_image(name_image, kind="name")
    stack = recognize_text_from_image(stack_image, kind="stack")
    active = is_player_name_active(name_image)

    dealer = False
    dealer_rect = regions.get("dealer")
    if dealer_rect is not None:
        dx0, dy0, dx1, dy1 = dealer_rect
        dealer_image = safe_crop(dx0, dy0, dx1, dy1)
        dealer = detect_dealer_marker(dealer_image)

    action = ""
    action_rect = regions.get("action")
    if action_rect is not None:
        ax0, ay0, ax1, ay1 = action_rect
        action_image = safe_crop(ax0, ay0, ax1, ay1)
        action_raw = recognize_table_action(action_image)
        action = resolve_table_action(action_raw)

    card_left = ""
    card_right = ""
    hole_cards = ""
    my_turn = False
    if seat_label == HERO_SEAT:
        right_rank_floor = ""
        for crop_key in HERO_HAND_OCR_ORDER:
            rect = regions.get(crop_key)
            if rect is None:
                continue
            if not isinstance(rect, tuple) or len(rect) != 4:
                continue
            x0, y0, x1, y1 = rect
            extra_image = safe_crop(x0, y0, x1, y1)
            raw = recognize_hole_card_crop(extra_image)
            if crop_key == "left_hand":
                rank = (
                    resolve_hole_card_rank_at_least(raw, right_rank_floor)
                    if right_rank_floor
                    else resolve_hole_card_rank(raw)
                )
                suit = dominant_hole_card_suit_color(extra_image)
                card_left = f"{rank}-{suit}" if rank else ""
                resolved = card_left
            elif crop_key == "right_hand":
                rank = resolve_hole_card_rank(raw)
                right_rank_floor = rank
                suit = dominant_hole_card_suit_color(extra_image)
                card_right = f"{rank}-{suit}" if rank else ""
                resolved = card_right
            elif crop_key == "hero_hand":
                hole_cards = normalize_hole_card_ranks(raw)
                resolved = hole_cards
            else:
                resolved = ""
            print(f"[{seat_label}] {crop_key}: OCR {raw!r} → {resolved!r}")

        my_turn_rect = regions.get("my_turn")
        if my_turn_rect is not None and isinstance(my_turn_rect, tuple) and len(my_turn_rect) == 4:
            mx0, my0, mx1, my1 = my_turn_rect
            my_turn_image = safe_crop(mx0, my0, mx1, my1)
            my_turn = detect_hero_my_turn_from_strip(my_turn_image)
            print(f"[{seat_label}] my_turn: {my_turn}")

    player = Player(
        seat=seat_label,
        name=player_name,
        stack=stack,
        active=active,
        dealer=dealer,
        action=action,
        card_left=card_left,
        card_right=card_right,
        hole_cards=hole_cards,
        my_turn=my_turn,
    )
    return player


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
    parser.add_argument(
        "--monitor",
        type=int,
        default=2,
        help="Display index for screencapture -D when capturing (default: 2 = first external).",
    )
    parser.add_argument(
        "--capture",
        action="store_true",
        help="Take a fresh monitor screenshot before OCR or --pick (default: use file on disk).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    source_path = (repo_root / args.image).resolve()

    if args.pick_both or args.pick:
        if args.capture:
            captured = screenshot_entire_monitor(
                source_path, monitor_index=args.monitor
            )
            print(f"Captured monitor to: {captured}")
        if args.pick_both:
            nx0, ny0, nx1, ny1 = _pick_and_print(source_path, "name")
            sx0, sy0, sx1, sy1 = _pick_and_print(source_path, "stack")
            print(
                'Paste under a seat key, e.g. "top_left": '
                f'{{ "name": {{"x0": {nx0}, "y0": {ny0}, "x1": {nx1}, "y1": {ny1}}}, '
                f'"stack": {{"x0": {sx0}, "y0": {sy0}, "x1": {sx1}, "y1": {sy1}}} }},'
            )
            return
        _pick_and_print(source_path, "region")
        print('Paste under a seat as "name" or "stack", e.g. top_left.name.')
        return

    config_path = repo_root / "crop_regions.json"
    source_rel, seats = load_crop_regions_config(config_path)
    image_path = repo_root / source_rel

    if args.capture:
        captured = screenshot_entire_monitor(image_path, monitor_index=args.monitor)
        print(f"Captured monitor to: {captured}")

    with Image.open(image_path) as monitor_image:
        base = monitor_image.copy()

    seat_items = list(seats.items())
    max_workers = min(8, len(seat_items)) if seat_items else 1
    crop_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(_process_one_seat, base, seat_label, regions, crop_lock)
            for seat_label, regions in seat_items
        ]
        players: list[Player] = [fut.result() for fut in futures]

    assign_six_max_positions(players)

    session = TableSessionTracker.load(DEFAULT_SESSION_PATH)
    session.ingest(players)
    saved_session = session.save(DEFAULT_SESSION_PATH)

    for player in players:
        print(f"[{player.seat}] Player: {player.name}")
        print(f"[{player.seat}] Stack: {player.stack}")
        print(f"[{player.seat}] Status: {player.status}")
        action_display = player.last_action if player.last_action else "—"
        print(f"[{player.seat}] Last action: {action_display}")
        print(f"[{player.seat}] Dealer: {player.dealer}")
        print(f"[{player.seat}] Position: {player.position}")
        action_show = player.action if player.action else "—"
        print(f"[{player.seat}] Action: {action_show}")
        if player.seat == HERO_SEAT:
            print(f"[{player.seat}] My turn: {'yes' if player.my_turn else 'no'}")
            if player.card_left or player.card_right or player.hole_cards:
                xy = player.hole_xy
                xy_part = f" ({xy})" if xy else ""
                print(
                    f"[{player.seat}] Hole cards — left: {player.card_left or '—'} | "
                    f"right: {player.card_right or '—'}{xy_part}"
                    + (
                        f" | combined: {player.hole_cards}"
                        if player.hole_cards
                        else ""
                    )
                )

    for line in session.report_lines():
        print(line)
    print(f"Saved hand session to: {saved_session}")


if __name__ == "__main__":
    main()
