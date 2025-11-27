"""
Grenade detection logic powered primarily by OCR.

The current approach extracts text inside the weapon HUD, runs Tesseract
to read the weapon name, and searches for known grenade keywords.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional
from subprocess import list2cmdline

import pytesseract
from pytesseract import Output

from config import MIN_CONFIDENCE, NADE_KEYWORDS, TESSERACT_CMD

# Configure pytesseract if a custom binary path is required.
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Restrict characters to reduce OCR noise.
CHAR_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz -"


def _debug_print_ocr(tokens: list[str], avg_conf: float, normalized: str) -> None:
    """Log the raw OCR output for debugging purposes."""
    if not tokens:
        print("[ocr] No tokens detected")
        return
    print(f"[ocr] tokens={tokens} avg_conf={avg_conf:.2f}")
    print(f"[ocr] normalized='{normalized}'")


def _build_tesseract_config(char_whitelist: str, psm: int = 6) -> str:
    """Return a Windows-safe Tesseract CLI string for pytesseract."""
    args = [
        "-c",
        f"tessedit_char_whitelist={char_whitelist}",
        "--psm",
        str(psm),
    ]
    return list2cmdline(args)


@dataclass
class NadeDetection:
    """Represents the most recent grenade classification."""

    label: str
    confidence: float
    raw_text: str


class NadeDetector:
    """
    Detects grenades by scanning OCR output for known keywords.

    If multiple keywords are present we pick the one with the highest
    individual confidence as reported by Tesseract's character-level data.
    """

    def __init__(
        self,
        keywords: Iterable[tuple[str, Iterable[str]]] = NADE_KEYWORDS,
        min_confidence: float = MIN_CONFIDENCE,
    ) -> None:
        self.keywords = tuple((label, tuple(variants)) for label, variants in keywords)
        self.min_confidence = min_confidence
        self.last_detection: Optional[NadeDetection] = None

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def _levenshtein(self, a: str, b: str) -> int:
        if len(a) < len(b):
            a, b = b, a
        previous_row = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            current_row = [i]
            for j, cb in enumerate(b, 1):
                insertions = previous_row[j] + 1
                deletions = current_row[j - 1] + 1
                substitutions = previous_row[j - 1] + (ca != cb)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def _search_keyword(self, text: str) -> Optional[str]:
        best_label = None
        best_distance = None
        for label, variants in self.keywords:
            for variant in variants:
                if variant in text:
                    return label
                distance = self._levenshtein(text, variant)
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_label = label
        if best_distance is not None and best_distance <= 2:
            return best_label
        return None

    def detect(self, gray_frame) -> Optional[NadeDetection]:
        """
        Run OCR on the processed HUD image and classify the held grenade.

        Returns None if the OCR confidence is too low or no grenade is found.
        """
        ocr_data = pytesseract.image_to_data(
            gray_frame,
            output_type=Output.DICT,
            config=_build_tesseract_config(CHAR_WHITELIST),
        )
        tokens = [token for token in ocr_data["text"] if token.strip()]
        confidences = [float(conf) for conf in ocr_data["conf"] if conf != "-1"]
        avg_conf = sum(confidences) / len(confidences) / 100 if confidences else 0.0

        normalized_text = self._normalize(" ".join(tokens))
        _debug_print_ocr(tokens, avg_conf, normalized_text)
        label = self._search_keyword(normalized_text)
        if not label or avg_conf < self.min_confidence:
            return None

        detection = NadeDetection(label=label, confidence=avg_conf, raw_text=normalized_text)
        self.last_detection = detection
        return detection

