"""Read crop regions from JSON, OCR — or interactively pick coordinates on the image."""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from argparse import Namespace

from PIL import Image

from consts import (
    HERO_HAND_OCR_ORDER,
    HERO_SEAT,
    normalize_hole_card_ranks,
    resolve_hole_card_rank,
    resolve_hole_card_rank_at_least,
    resolve_table_action,
)
from hand_session import (
    DEFAULT_SESSION_PATH,
    TableSessionTracker,
    build_table_snapshot,
    preflop_last_aggressor_before_hero,
    preflop_spot_for_hero,
    preflop_situation_before_hero,
)
from player import Player, assign_six_max_positions
from street_hash import (
    format_u64_hash,
    is_preflop_street_crop,
    street_crop_average_hash_u64,
)
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


def _run_ocr_cycle(
    args: Namespace,
    *,
    session: TableSessionTracker,
    config_path: Path,
    config_data: dict,
    seats: dict,
    image_path: Path,
    iteration: int | None = None,
) -> tuple[bool, str, dict[str, float]]:
    """One capture + OCR + session update.

    Returns ``(stop_watch, reason, timings_ms)``. ``stop_watch`` is True when hero
    should take over (``my_turn``) or hero has folded (seat ``Inactive``).
    """
    t_start = perf_counter()
    timings_ms: dict[str, float] = {}

    if iteration is not None:
        print(f"\n=== watch iteration {iteration} ===")

    hero_was_on_turn = session.saved_hero_my_turn()
    hero_was_active_before = session.saved_hero_seat_active()
    # While polling until our turn, always grab a fresh frame (unless --no-capture).
    do_capture = not args.no_capture and (
        bool(getattr(args, "watch_until_my_turn", False)) or not hero_was_on_turn
    )

    t_cap0 = perf_counter()
    if hero_was_on_turn and not getattr(args, "watch_until_my_turn", False):
        print("Skipping live capture: hero was on turn in last saved table_snapshot.")
    if do_capture:
        captured = screenshot_entire_monitor(image_path, monitor_index=args.monitor)
        print(f"Captured monitor to: {captured}")
    timings_ms["capture"] = (perf_counter() - t_cap0) * 1000.0

    t_img0 = perf_counter()
    with Image.open(image_path) as monitor_image:
        base = monitor_image.copy()
    timings_ms["image_open_copy"] = (perf_counter() - t_img0) * 1000.0

    t_street0 = perf_counter()
    street_region = config_data.get("street")
    if isinstance(street_region, dict) and all(
        key in street_region for key in ("x0", "y0", "x1", "y1")
    ):
        sx0 = int(street_region["x0"])
        sy0 = int(street_region["y0"])
        sx1 = int(street_region["x1"])
        sy1 = int(street_region["y1"])
        street_image = crop_from_loaded_image(base, sx0, sy0, sx1, sy1)

        cur_u64 = street_crop_average_hash_u64(street_image)
        street_h_hex = format_u64_hash(cur_u64)
        print(f"Street crop average-hash (64-bit): {street_h_hex}")

        if args.record_street_preflop_hash:
            config_data["street_preflop_ahash"] = street_h_hex
            config_data.setdefault("street_preflop_hamming_max", 10)
            config_path.write_text(
                json.dumps(config_data, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(
                f"Recorded preflop street hash to {config_path} "
                f"(street_preflop_hamming_max={config_data['street_preflop_hamming_max']})."
            )
        else:
            ref_hex = config_data.get("street_preflop_ahash")
            try:
                hmax = int(config_data.get("street_preflop_hamming_max", 10))
            except (TypeError, ValueError):
                hmax = 10
            try:
                is_pf, dist, _u = is_preflop_street_crop(
                    street_image,
                    ref_hex=ref_hex if isinstance(ref_hex, str) else None,
                    hamming_max=hmax,
                )
            except ValueError as exc:
                print(f"Street vs preflop: bad street_preflop_ahash in config ({exc}).")
            else:
                if is_pf is None:
                    print(
                        "Street vs preflop: no reference (set street_preflop_ahash or run "
                        "with --record-street-preflop-hash on a known-preflop screen)."
                    )
                else:
                    print(
                        f"Street vs preflop: {'yes' if is_pf else 'no'} "
                        f"(Hamming={dist}, max={hmax}, ref={ref_hex})."
                    )
    timings_ms["street_hash"] = (perf_counter() - t_street0) * 1000.0

    t_ocr0 = perf_counter()
    seat_items = list(seats.items())
    max_workers = 6
    crop_lock = threading.Lock()
    print(f"Seat OCR workers: {max_workers} (seats={len(seat_items)})")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(_process_one_seat, base, seat_label, regions, crop_lock)
            for seat_label, regions in seat_items
        ]
        players: list[Player] = [fut.result() for fut in futures]
    timings_ms["seat_ocr"] = (perf_counter() - t_ocr0) * 1000.0

    t_post0 = perf_counter()
    assign_six_max_positions(players)

    hero = next((p for p in players if p.seat == HERO_SEAT), None)
    situation: str | None = None
    spot: str | None = None
    last_aggressor: str | None = None
    if hero is not None and hero.my_turn:
        situation = preflop_situation_before_hero(players)
        spot = preflop_spot_for_hero(players)
        la = preflop_last_aggressor_before_hero(players)
        last_aggressor = la or None

    snapshot = build_table_snapshot(
        players,
        situation=situation,
        spot=spot,
        last_aggressor=last_aggressor,
    )
    snapshot_changed = session.set_table_snapshot_if_changed(snapshot)
    if snapshot_changed:
        session.ingest(players)
    else:
        print("Table snapshot unchanged; skipped hand_session ingest/update.")

    hero_folded = hero is not None and not hero.active
    fold_transition = hero_folded and hero_was_active_before
    should_save = (
        snapshot_changed
        or (hero is not None and hero.my_turn and not hero_was_on_turn)
        or fold_transition
    )
    if should_save:
        saved_session = session.save(DEFAULT_SESSION_PATH)
        print(f"Saved hand session to: {saved_session}")
    else:
        print(
            "Hand session file not written (snapshot unchanged; "
            "already saved this state, or no hero turn/fold transition to persist)."
        )

    timings_ms["postprocess_and_session_save"] = (perf_counter() - t_post0) * 1000.0

    # On transition into hero turn, open matching preflop strategy image once.
    if hero is not None and hero.my_turn and not hero_was_on_turn and spot and last_aggressor:
        strategy_path = (
            config_path.parent
            / "preflop_strategy_images"
            / hero.position
            / spot
            / f"{last_aggressor}.png"
        )
        if strategy_path.exists():
            print(f"Opening strategy image: {strategy_path}")
            subprocess.run(["open", str(strategy_path)], check=False)
        else:
            print(f"Strategy image not found: {strategy_path}")

    t_print0 = perf_counter()
    hero_position = hero.position if hero is not None else "?"
    print(f"[hero] Position: {hero_position}")
    print(f"[hero] My turn: {'yes' if bool(hero and hero.my_turn) else 'no'}")
    if hero is not None and hero.my_turn:
        if spot:
            print(f"[hero] Spot: {spot}")
        if last_aggressor and spot and spot != "RFI":
            print(f"[hero] Last aggressor: {last_aggressor}")
        if situation:
            print(f"[hero] Situation: {situation}")

    for line in session.report_lines():
        print(line)
    timings_ms["print_output"] = (perf_counter() - t_print0) * 1000.0
    timings_ms["total"] = (perf_counter() - t_start) * 1000.0
    print(
        "Runtime (ms): "
        + ", ".join(f"{k}={v:.1f}" for k, v in timings_ms.items())
    )

    if hero is not None and hero.my_turn:
        return True, "my_turn", timings_ms
    if hero is not None and not hero.active:
        return True, "folded", timings_ms
    return False, "", timings_ms


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
        "--no-capture",
        action="store_true",
        help="Do not call screencapture; use the existing screenshot file on disk.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single capture + OCR pass and exit (default is watch until hero's turn).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between iterations in default watch mode (default: 1.0).",
    )
    parser.add_argument(
        "--record-street-preflop-hash",
        action="store_true",
        help="After cropping ``street``, write 64-bit average-hash reference into crop_regions.json.",
    )
    args = parser.parse_args()
    # Default: poll until it is hero's turn. ``_run_ocr_cycle`` uses this to force fresh captures.
    args.watch_until_my_turn = not args.once

    repo_root = Path(__file__).resolve().parent
    source_path = (repo_root / args.image).resolve()

    if args.pick_both or args.pick:
        if not args.no_capture:
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
    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    source_rel, seats = load_crop_regions_config(config_path)
    image_path = repo_root / source_rel

    session = TableSessionTracker.load(DEFAULT_SESSION_PATH)

    if args.watch_until_my_turn:
        print(
            "Watch mode (default): capture until hero's turn or hero folds; "
            f"interval={args.interval}s (Ctrl+C to abort). "
            "Pass --once for a single pass. "
            "Use --no-capture to poll the same file instead (testing only)."
        )
        n = 0
        while True:
            n += 1
            stop, reason, _timings = _run_ocr_cycle(
                args,
                session=session,
                config_path=config_path,
                config_data=config_data,
                seats=seats,
                image_path=image_path,
                iteration=n,
            )
            if stop:
                if reason == "my_turn":
                    print("Stopping watch: hero's turn detected.")
                elif reason == "folded":
                    print("Stopping watch: hero folded (seat inactive).")
                else:
                    print("Stopping watch.")
                break
            time.sleep(max(0.0, args.interval))
    else:
        _stop, _reason, _timings = _run_ocr_cycle(
            args,
            session=session,
            config_path=config_path,
            config_data=config_data,
            seats=seats,
            image_path=image_path,
            iteration=None,
        )


if __name__ == "__main__":
    main()
