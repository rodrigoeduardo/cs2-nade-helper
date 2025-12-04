"""
User feedback helpers.

For the prototype we emit desktop notifications through `plyer` whenever
the detected grenade changes. This keeps the UI simple while still making
the state visible even when CS2 is focused.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

from PIL import Image, ImageTk
import tkinter as tk
from plyer import notification

from .config import DEBUG_SAVE_RADAR_PATH, DEBUG_SAVE_RADAR_NORMALIZED_PATH, DEBUG_SAVE_NORTH_DETECTION_PATH, DEBUG_SAVE_PLAYER_POS_PATH

@dataclass
class OverlayState:
    message: str
    detail: str
    timestamp: float


class NotificationOverlay:
    """Lightweight notifier for grenade detection events with GUI overlay."""

    def __init__(self, title: str = "CS2 Nade Helper", min_interval: float = 1.5) -> None:
        self.title = title
        self.min_interval = min_interval
        self.last_state: Optional[OverlayState] = None
        
        # GUI window
        self.root: Optional[tk.Tk] = None
        self.window_created = False
        self.current_message = "No grenade detected"
        self.current_detail = ""
        
        # Image paths
        self.debug_images = {
            "radar": DEBUG_SAVE_RADAR_PATH,
            "radar_normalized": DEBUG_SAVE_RADAR_NORMALIZED_PATH,
            "player_pos": DEBUG_SAVE_PLAYER_POS_PATH,
            "north_detection": DEBUG_SAVE_NORTH_DETECTION_PATH
        }
        
        # Store PhotoImage references to prevent garbage collection
        self.image_photos: dict[str, ImageTk.PhotoImage] = {}
        
        # Start GUI in a separate thread
        self._init_gui()

    def _init_gui(self) -> None:
        """Initialize the GUI window in a separate thread."""
        def create_window():
            self.root = tk.Tk()
            self.root.title(self.title)
            self.root.geometry("1200x800")
            self.root.resizable(True, True)
            # Make window appear on top initially
            self.root.lift()
            self.root.attributes('-topmost', True)
            self.root.after_idle(lambda: self.root.attributes('-topmost', False))
            
            # Main frame
            main_frame = tk.Frame(self.root)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Current nade label
            nade_frame = tk.Frame(main_frame)
            nade_frame.pack(fill=tk.X, pady=(0, 10))
            
            tk.Label(nade_frame, text="Current Nade:", font=("Arial", 14, "bold")).pack(side=tk.LEFT)
            self.nade_label = tk.Label(nade_frame, text=self.current_message, font=("Arial", 16), fg="blue")
            self.nade_label.pack(side=tk.LEFT, padx=(10, 0))
            
            # Detail label
            self.detail_label = tk.Label(main_frame, text=self.current_detail, font=("Arial", 10), fg="gray")
            self.detail_label.pack(fill=tk.X, pady=(0, 10))
            
            # Images frame
            images_frame = tk.Frame(main_frame)
            images_frame.pack(fill=tk.BOTH, expand=True)
            
            # Create canvas widgets for each debug image
            self.image_canvases = {}
            self.image_photos = {}  # Keep references to PhotoImage objects
            image_keys = list(self.debug_images.keys())
            for i, key in enumerate(image_keys):
                # Create frame for each image
                img_frame = tk.Frame(images_frame, relief=tk.RAISED, borderwidth=1)
                row = i // 2
                col = i % 2
                img_frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
                
                # Title
                title_label = tk.Label(img_frame, text=key.replace("_", " ").title(), font=("Arial", 10, "bold"))
                title_label.pack(pady=(5, 2))
                
                # Canvas for image display (better for scaling)
                canvas = tk.Canvas(img_frame, bg="lightgray", width=550, height=450)
                canvas.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
                self.image_canvases[key] = canvas
                
                # Configure grid weights for resizing
                images_frame.grid_columnconfigure(col, weight=1)
                images_frame.grid_rowconfigure(row, weight=1)
            
            # Update images periodically
            self._update_images()
            
            # Handle window close
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)
            
            self.window_created = True
            self.root.mainloop()
        
        # Start GUI thread
        gui_thread = threading.Thread(target=create_window, daemon=True)
        gui_thread.start()
        
        # Wait a bit for window to be created
        time.sleep(0.5)

    def _update_images(self) -> None:
        """Update the debug images in the GUI."""
        if not self.root:
            return
            
        for key, filename in self.debug_images.items():
            canvas = self.image_canvases.get(key)
            if not canvas:
                continue
                
            # Get canvas size first
            canvas.update_idletasks()
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            
            # Use actual canvas size or default if not yet rendered
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width, canvas_height = 550, 450
            
            if os.path.exists(filename):
                try:
                    # Load image
                    img = Image.open(filename)
                    
                    # Resize image to fit canvas while maintaining aspect ratio
                    img.thumbnail((canvas_width - 10, canvas_height - 10), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    
                    # Clear canvas and add image centered
                    canvas.delete("all")
                    # Center the image
                    x = (canvas_width - img.width) // 2
                    y = (canvas_height - img.height) // 2
                    canvas.create_image(x, y, anchor=tk.NW, image=photo)
                    
                    # Keep a reference to prevent garbage collection
                    self.image_photos[key] = photo
                except Exception as e:
                    print(f"[overlay] Failed to load {filename}: {e}")
                    canvas.delete("all")
                    canvas.create_text(canvas_width // 2, canvas_height // 2, 
                                     text=f"Error loading image", fill="red")
            else:
                canvas.delete("all")
                canvas.create_text(canvas_width // 2, canvas_height // 2, 
                                 text="Image not available", fill="gray")
        
        # Schedule next update
        if self.root:
            self.root.after(500, self._update_images)  # Update every 500ms

    def _on_close(self) -> None:
        """Handle window close event."""
        # Don't actually close, just minimize
        self.root.iconify()

    def _should_notify(self, message: str) -> bool:
        if not self.last_state:
            return True
        if self.last_state.message == message:
            return False
        return (time.time() - self.last_state.timestamp) >= self.min_interval

    def show(self, message: str, detail: str = "") -> None:
        """Send a desktop notification if enough time passed or the message changed, and update GUI."""
        # Update GUI
        self.current_message = message
        self.current_detail = detail
        
        if self.root:
            try:
                self.root.after(0, lambda: self._update_gui_text())
            except Exception:
                pass  # Window might be closing
        
        # Send notification if needed
        if not self._should_notify(message):
            return
        notification.notify(title=self.title, message=f"{message}\n{detail}".strip(), timeout=2)
        self.last_state = OverlayState(message=message, detail=detail, timestamp=time.time())

    def _update_gui_text(self) -> None:
        """Update the GUI text labels (must be called from main thread)."""
        if self.root and hasattr(self, 'nade_label'):
            self.nade_label.config(text=self.current_message)
            self.detail_label.config(text=self.current_detail)

