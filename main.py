"""Main loop for V2 detection and dodge logic."""

import argparse
import json
import logging
import os
import time

from adb_controller import ADBController
from debug_overlay import save_debug_frame
from dodge_planner import DodgePlanner
from state_machine import DodgeStateMachine
from threat_detector import ThreatDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Auto-dodge-brawl-stars V2")
    parser.add_argument("--device", type=str, help="ADB serial")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device:
        config["device_id"] = args.device
    if args.dry_run:
        config["dry_run"] = True
    if args.debug:
        config["debug"] = True

    output_dir = config.get("debug_output_dir", "debug")
    if config.get("debug"):
        os.makedirs(output_dir, exist_ok=True)

    controller = ADBController(config.get("device_id", ""))
    if not controller.connect():
        logger.error("Cannot connect to device. Check ADB and USB authorization.")
        return

    planner = DodgePlanner(dodge_range=config.get("dodge_range", 320))
    tracker = ThreatDetector(
        min_contour_area=config.get("min_contour_area", 45),
        min_motion_pixels=config.get("min_motion_pixels", 3),
        max_tracking_jump=config.get("max_tracking_jump", 180),
        min_threat_score=config.get("min_threat_score", 0.42),
        threat_confirmation_frames=config.get("threat_confirmation_frames", 2),
        max_candidate_distance=config.get("max_candidate_distance", 0.48),
        roi=config.get("roi", {"left": 0.08, "top": 0.10, "right": 0.92, "bottom": 0.92}),
    )
    state = DodgeStateMachine(dodge_cooldown_seconds=config.get("detection_cooldown_seconds", 0.2))

    logger.info("Connected. Starting detection loop...")
    frame_count = 0
    dodge_count = 0
    start_time = time.time()

    try:
        while True:
            screenshot = controller.get_screenshot()
            if screenshot is None:
                time.sleep(0.5)
                continue

            frame_count += 1
            width, height = screenshot.size
            player_position = (width // 2, height // 2)
            threat = tracker.update(screenshot, player_position)
            debug_info = tracker.get_last_debug()
            state.update(threat is not None)
            dodge_target = None

            if threat is not None:
                dodge_target = planner.plan(player_position, threat, (width, height))
                if config.get("debug"):
                    logger.info(
                        "Threat at %s | candidates=%s | score=%s | confirmations=%s | dodge_target=%s",
                        threat,
                        len(debug_info.get("candidates", [])),
                        debug_info.get("best"),
                        debug_info.get("confirmation_count"),
                        dodge_target,
                    )
                if state.can_dodge():
                    if config.get("dry_run"):
                        logger.info("DRY RUN: dodge would go to %s", dodge_target)
                    else:
                        controller.swipe(
                            player_position[0], player_position[1],
                            dodge_target[0], dodge_target[1],
                            duration_ms=config.get("dodge_duration_ms", 220),
                        )
                        dodge_count += 1
                        logger.info("Dodge executed #%s to %s", dodge_count, dodge_target)
                    state.on_dodge()

            if config.get("debug"):
                debug_path = save_debug_frame(
                    screenshot, player_position, debug_info.get("candidates", []),
                    threat, dodge_target, output_dir=output_dir,
                )
                logger.info("Debug frame saved: %s", debug_path)

            if frame_count % 100 == 0:
                elapsed = max(0.001, time.time() - start_time)
                logger.info("Stats - FPS: %.1f | Dodges: %s | Frames: %s", frame_count / elapsed, dodge_count, frame_count)
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        controller.disconnect()
        logger.info("Session ended.")


if __name__ == "__main__":
    main()
