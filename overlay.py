"""
User feedback helpers.

For the prototype we emit desktop notifications through `plyer` whenever
the detected grenade changes. This keeps the UI simple while still making
the state visible even when CS2 is focused.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from plyer import notification


@dataclass
class OverlayState:
    message: str
    detail: str
    timestamp: float


class NotificationOverlay:
    """Lightweight notifier for grenade detection events."""

    def __init__(self, title: str = "CS2 Nade Helper", min_interval: float = 1.5) -> None:
        self.title = title
        self.min_interval = min_interval
        self.last_state: Optional[OverlayState] = None

    def _should_notify(self, message: str) -> bool:
        if not self.last_state:
            return True
        if self.last_state.message == message:
            return False
        return (time.time() - self.last_state.timestamp) >= self.min_interval

    def show(self, message: str, detail: str = "") -> None:
        """Send a desktop notification if enough time passed or the message changed."""
        if not self._should_notify(message):
            return
        notification.notify(title=self.title, message=f"{message}\n{detail}".strip(), timeout=2)
        self.last_state = OverlayState(message=message, detail=detail, timestamp=time.time())

