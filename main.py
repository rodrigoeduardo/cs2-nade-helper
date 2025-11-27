"""
Entrypoint for the CS2 nade helper prototype.

Current functionality:
- Grab the weapon HUD via MSS.
- Run OCR to detect which grenade is currently equipped.
- Emit desktop notifications summarizing the detection.
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


class NadeHelperApp:
    def __init__(self, interval: float = CAPTURE_INTERVAL_SECONDS) -> None:
        self.interval = interval
        self.capture = HUDCapture()
        self.detector = NadeDetector()
        self.overlay = NotificationOverlay()
        self._stop_event = threading.Event()
        self.current_detection: Optional[NadeDetection] = None

    def stop(self) -> None:
        self._stop_event.set()

    def _handle_detection(self, detection: Optional[NadeDetection]) -> None:
        if detection:
            self.current_detection = detection

        current = self.current_detection
        if current:
            msg = current.label.title()
            detail = f"OCR confidence: {current.confidence:.0%}\nRaw: {current.raw_text}"
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
            time.sleep(self.interval)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CS2 nade helper prototype")
    parser.add_argument(
        "--interval",
        type=float,
        default=CAPTURE_INTERVAL_SECONDS,
        help="Capture interval in seconds (default: config value)",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    app = NadeHelperApp(interval=args.interval)

    def handle_signal(signum, frame):  # pragma: no cover - signal handling
        print(f"\nReceived signal {signum}, stopping...")
        app.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print("Nade helper running. Press Ctrl+C to exit.")
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
