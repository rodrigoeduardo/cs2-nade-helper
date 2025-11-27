"""
Centralized runtime configuration for the CS2 nade helper prototype.

The actual screen coordinates will depend on the player's resolution and
HUD scale. The defaults below assume 1920x1080 with default HUD size and
should be treated as placeholders that the user customizes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ScreenRegion:
    """Simple rectangle defined in absolute screen coordinates."""

    top: int
    left: int
    width: int
    height: int

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def right(self) -> int:
        return self.left + self.width

    def as_mss_dict(self) -> dict:
        return {"top": self.top, "left": self.left, "width": self.width, "height": self.height}


# Monitor index used by mss (1 = primary, 2 = secondary, etc.).
CAPTURE_MONITOR_INDEX: int = 1

# Placeholder HUD region covering the weapon name text (bottom-right corner).
HUD_REGION = ScreenRegion(top=1055, left=1650, width=270, height=25)

# Preprocessing controls for HUDCapture.
PREPROCESS_SCALE: float = 5
PREPROCESS_USE_CLAHE: bool = True
PREPROCESS_ADAPTIVE_BLOCK_SIZE: int = 25  # Must be odd.
PREPROCESS_ADAPTIVE_C: int = 9

# When True, save the processed grayscale frame to disk each capture loop.
DEBUG_SAVE_PROCESSED: bool = False
DEBUG_SAVE_PATH: str = "debug_thresh.png"
DEBUG_SAVE_GRAY_PATH: str = "debug_gray.png"

# Polling interval (seconds) between capture attempts.
CAPTURE_INTERVAL_SECONDS: float = 0.35

# Minimum confidence (0-1) before we accept a detected grenade keyword.
MIN_CONFIDENCE: float = 0.05

# Pairs of canonical grenade labels and the substrings we expect from OCR.
NADE_KEYWORDS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("smoke", ("smoke", "granada de fumaca", "fumaca")),
    ("molotov", ("molotov", "granada incendiaria", "incendiaria")),
    ("flashbang", ("flashbang", "granada de luz", "luz")),
    ("he", ("explosive", "granada explosiva", "explosiva")),
)

# Optional path to the tesseract executable. Leave as None to rely on PATH.
TESSERACT_CMD: str | None = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
