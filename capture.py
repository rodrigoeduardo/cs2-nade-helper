"""
Screen capture utilities responsible for grabbing the CS2 HUD weapon panel.

The module uses `mss` for high-performance screen capture and OpenCV for
basic image preprocessing that improves OCR robustness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from mss import mss

from config import (
    CAPTURE_MONITOR_INDEX,
    DEBUG_SAVE_GRAY_PATH,
    DEBUG_SAVE_PATH,
    DEBUG_SAVE_PROCESSED,
    HUD_REGION,
    PREPROCESS_ADAPTIVE_BLOCK_SIZE,
    PREPROCESS_ADAPTIVE_C,
    PREPROCESS_SCALE,
    PREPROCESS_USE_CLAHE,
    ScreenRegion,
)


@dataclass
class CaptureResult:
    """Container that keeps both the raw and preprocessed frames."""

    raw_bgra: np.ndarray
    processed_gray: np.ndarray


class HUDCapture:
    """Grabs the configured HUD region and prepares it for OCR."""

    def __init__(self, region: ScreenRegion = HUD_REGION, scale: float = 1.5) -> None:
        self.region = region
        self.sct = mss()
        self.scale = scale

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert BGRA frame to a denoised, high-contrast grayscale image.

        The weapon name uses bright text over a darker background, so a
        simple adaptive threshold after resizing dramatically improves OCR.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        if self.scale != 1.0:
            new_size = (int(gray.shape[1] * self.scale), int(gray.shape[0] * self.scale))
            gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_CUBIC)

        if PREPROCESS_USE_CLAHE:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        block_size = max(3, PREPROCESS_ADAPTIVE_BLOCK_SIZE | 1)
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            PREPROCESS_ADAPTIVE_C,
        )
        return gray, thresh

    def grab(self) -> Optional[CaptureResult]:
        """Capture the configured region; returns None on failure."""
        region = self.region.as_mss_dict()
        region["mon"] = CAPTURE_MONITOR_INDEX

        try:
            shot = self.sct.grab(region)
        except Exception as exc:  # pragma: no cover - hardware specific
            print(f"[capture] Failed to grab HUD region: {exc}")
            return None

        frame = np.array(shot)
        gray, processed = self._preprocess(frame)

        # if DEBUG_SAVE_PROCESSED:
        #     cv2.imwrite(DEBUG_SAVE_GRAY_PATH, gray)
        #     cv2.imwrite(DEBUG_SAVE_PATH, processed)
        #     print(f"[capture] Saved debug frames to {DEBUG_SAVE_GRAY_PATH}, {DEBUG_SAVE_PATH}")

        return CaptureResult(raw_bgra=frame, processed_gray=gray)
