"""
Entrypoint for the CS2 nade helper prototype.

Current functionality:
- Grab the weapon HUD via MSS.
- Run OCR to detect which grenade is currently equipped.
- Emit desktop notifications summarizing the detection.
- (New) Opens browser with lineups for detected nade.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from typing import Optional

from capture import HUDCapture
from config import CAPTURE_INTERVAL_SECONDS
from detection import NadeDetector, NadeDetection
from overlay import NotificationOverlay
from browser import BrowserController
from radar import RadarMatcher


class NadeHelperApp:
    def __init__(self, interval: float = CAPTURE_INTERVAL_SECONDS, browser_enabled: bool = False) -> None:
        self.interval = interval
        self.capture = HUDCapture()
        self.detector = NadeDetector()
        self.overlay = NotificationOverlay()
        self._stop_event = threading.Event()
        self.current_detection: Optional[NadeDetection] = None
        
        self.browser_enabled = browser_enabled
        self.browser: Optional[BrowserController] = None
        if browser_enabled:
            self.browser = BrowserController()
            self.radar = RadarMatcher()
            
        self.last_navigated_nade: Optional[str] = None
        self.last_pos_update: float = 0
        self.pos_update_interval: float = 2.0  # Update map position every 2 seconds

    def stop(self) -> None:
        """Stop the application and clean up resources."""
        self._stop_event.set()
        if self.browser:
            try:
                self.browser.close()
            except Exception as e:
                print(f"[main] Error closing browser: {e}")

    def _handle_detection(self, detection: Optional[NadeDetection]) -> None:
        if detection:
            self.current_detection = detection

        current = self.current_detection
        if current:
            msg = current.label.title()
            # detail = f"OCR confidence: {current.confidence:.0%}\nRaw: {current.raw_text}"
            detail = ""
            
            # Browser navigation logic
            if self.browser:
                # 1. Navigate if nade type changed
                if current.label != self.last_navigated_nade:
                    # TODO: Get map from radar detection. Hardcoded to 'mirage' for now.
                    self.browser.navigate("de_mirage", current.label)
                    self.last_navigated_nade = current.label

                # 2. Update player position on map (throttled)
                now = time.time()
                if now - self.last_pos_update > self.pos_update_interval:
                    pos = self.radar.find_position("de_mirage")
                    if pos:
                        rx, ry = pos
                        self.browser.click_map_position(rx, ry)
                        self.last_pos_update = now

        else:
            msg = "No grenade detected"
            detail = "Keep the grenade wheel open for clearer text."
        
        self.overlay.show(msg, detail)
        print(f"Current nade: {msg}", detail)

    def loop_once(self) -> None:
        result = self.capture.grab()
        if not result:
            self.overlay.show(
                "Capture error", "Check HUD coordinates in config.py")
            print("Capture error", "Check HUD coordinates in config.py")
            return
        detection = self.detector.detect(result.processed_gray)
        self._handle_detection(detection)

    def run(self) -> None:
        while not self._stop_event.is_set():
            self.loop_once()
            # Use wait() instead of sleep() so it can be interrupted immediately
            self._stop_event.wait(timeout=self.interval)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CS2 nade helper prototype")
    parser.add_argument(
        "--interval",
        type=float,
        default=CAPTURE_INTERVAL_SECONDS,
        help="Capture interval in seconds (default: config value)",
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Enable browser automation for lineups",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    app = NadeHelperApp(interval=args.interval, browser_enabled=args.browser)

    def handle_signal(signum, frame):  # pragma: no cover - signal handling
        print(f"\nReceived signal {signum}, stopping...")
        app.stop()
        # Force exit if cleanup takes too long
        import sys
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print("Nade helper running. Press Ctrl+C to exit.")
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
