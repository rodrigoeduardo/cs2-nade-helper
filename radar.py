"""
Radar capture logic to identify player position on the map using Feature Matching.
"""

import cv2
import numpy as np
import os
from typing import Optional, Tuple
from mss import mss

from config import CAPTURE_MONITOR_INDEX, RADAR_REGION, DEBUG_SAVE_PROCESSED, DEBUG_SAVE_RADAR_PATH 

class RadarMatcher:
    """
    Matches the current in-game radar view against a full map overview
    to determine the player's position.
    """
    
    def __init__(self, maps_dir: str = "maps") -> None:
        self.maps_dir = maps_dir
        self.sct = mss()
        
        # Feature detector (ORB is fast and rotation invariant)
        self.orb = cv2.ORB_create(nfeatures=1000)
        
        # Cache for loaded maps: {map_name: (image, keypoints, descriptors, padding, scale_factor)}
        # padding = (pad_x, pad_y) added to match game radar black borders at edges
        # scale_factor = calculated based on map dimensions relative to mirage calibration
        self._cache = {}
        
        # Padding: fixed 800px on each side to handle edge cases
        # When player is at map borders, radar shows black areas outside the map
        self.padding_px = 800
        
        # Calibration constants (from mirage map)
        # Mirage: 1374x1196px, scale factor 3.57 works perfectly
        self.MIRAGE_AVG_DIMENSION = (1374 + 1196) / 2.0  # 1285px
        self.MIRAGE_SCALE_FACTOR = 3.57
        
        # Debug mode
        self.debug = False

    def _load_map(self, map_name: str) -> bool:
        """Load map reference image, add black padding, and pre-calculate features."""
        if map_name in self._cache:
            return True
            
        # Try without _radar suffix first, as per user update
        filename = f"{map_name}.png"
        path = os.path.join(self.maps_dir, filename)
        
        if not os.path.exists(path):
            # Fallback to old naming if needed, or fail
            filename_legacy = f"{map_name}_radar.png"
            path_legacy = os.path.join(self.maps_dir, filename_legacy)
            if os.path.exists(path_legacy):
                path = path_legacy
            else:
                print(f"[radar] Map file not found: {path}")
                return False
            
        # Load image in grayscale
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"[radar] Failed to load image: {path}")
            return False
        
        # Add fixed black padding to handle edge cases
        # When player is at map borders, the radar shows black areas outside the map
        # 800px padding ensures matching works even at map corners
        h, w = img.shape
        pad_x = self.padding_px
        pad_y = self.padding_px
        
        padded_img = cv2.copyMakeBorder(
            img, 
            top=pad_y, bottom=pad_y, 
            left=pad_x, right=pad_x, 
            borderType=cv2.BORDER_CONSTANT, 
            value=0  # Black padding
        )
        
        # Calculate scale factor based on map dimensions
        # Scale proportionally relative to mirage calibration
        map_avg_dimension = (w + h) / 2.0
        scale_factor = self.MIRAGE_SCALE_FACTOR * (map_avg_dimension / self.MIRAGE_AVG_DIMENSION)
            
        # Detect features on padded image (kept for potential future use)
        kp, des = self.orb.detectAndCompute(padded_img, None)
        
        self._cache[map_name] = (padded_img, kp, des, (pad_x, pad_y), scale_factor)
        print(f"[radar] Loaded {map_name} ({w}x{h}) with {pad_x}px padding, scale factor: {scale_factor:.3f}")
        return True

    def grab_radar(self) -> np.ndarray | None:
        """Capture the radar region."""
        region = RADAR_REGION.as_mss_dict()
        region["mon"] = CAPTURE_MONITOR_INDEX
        try:
            shot = self.sct.grab(region)
            # Convert to grayscale directly
            img = np.array(shot)
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            if DEBUG_SAVE_PROCESSED:
                cv2.imwrite(DEBUG_SAVE_RADAR_PATH, gray)
                print(f"[capture] Saved debug frames to {DEBUG_SAVE_RADAR_PATH}")
            return gray
        except Exception as exc:
            print(f"[radar] Failed to grab radar: {exc}")
            return None

    def _mask_radar(self, gray_frame: np.ndarray) -> np.ndarray:
        """
        Apply a circular mask to keep only the radar content.
        Removes HUD elements outside the circle which might confuse matching.
        """
        h, w = gray_frame.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        radius = min(w, h) // 2 - 10  # Slightly smaller to cut off border
        
        cv2.circle(mask, center, radius, (255), -1)
        masked = cv2.bitwise_and(gray_frame, gray_frame, mask=mask)
        return masked

    def _detect_north_angle(self, gray_frame: np.ndarray) -> float:
        """
        Detect the North indicator (white triangle) around the radar circle.
        Returns the angle in degrees where North is located (0 = top, clockwise positive).
        
        Geometry (based on 300x300 capture region):
        - Capture region: 300x300px
        - Radar circle: 250x250px centered (radius 125px, center at 150,150)
        - North triangle: starts 10px outside radar circle edge (at radius ~135px)
        - Triangle height: ~10px (extends to radius ~145px)
        - Triangle color: #ffffff (pure white = 255 in grayscale)
        
        The sampling scales proportionally if image size differs from 300x300.
        """
        h, w = gray_frame.shape
        center_x, center_y = w // 2, h // 2
        
        # Scale factor in case image is not exactly 300x300
        scale = min(w, h) / 300.0
        
        # Radar circle radius is 125px at 300x300 (250px diameter / 2)
        radar_radius = int(125 * scale)
        
        # The north triangle is ~10px outside the radar circle
        # and has ~10px height, so sample from just outside radar to edge of triangle
        inner_sample_radius = int(radar_radius + 8 * scale)   # ~133px - where triangle starts
        outer_sample_radius = int(radar_radius + 20 * scale)  # ~145px - outer edge of triangle
        
        # White threshold - north indicator is pure white (#ffffff = 255)
        # Use a high threshold to filter out gray map content
        WHITE_THRESHOLD = 250
        
        # Calibration offset (degrees) to correct for triangle shape bias
        # The triangle points toward center, so detection tends to find the
        # center of mass rather than the tip. Positive = clockwise adjustment.
        NORTH_CALIBRATION_OFFSET = 7.0
        
        # Sample 360 points around the circle
        num_samples = 360
        angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
        
        white_pixel_counts = []
        for angle in angles:
            # Count white pixels at this angle across the sampling ring
            white_count = 0
            for r in range(inner_sample_radius, outer_sample_radius + 1):
                # Angle convention: 0 = top, 90 = right, 180 = bottom, 270 = left
                x = int(center_x + r * np.sin(angle))
                y = int(center_y - r * np.cos(angle))
                
                if 0 <= x < w and 0 <= y < h:
                    if gray_frame[y, x] >= WHITE_THRESHOLD:
                        white_count += 1
            
            white_pixel_counts.append(white_count)
        
        white_pixel_counts = np.array(white_pixel_counts)
        
        # Smooth to find the center of the triangle (which spans ~10-15 degrees)
        kernel_size = 15
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(white_pixel_counts, kernel, mode='same')
        
        # Find the angle with the most white pixels (center of the triangle)
        north_idx = np.argmax(smoothed)
        north_angle_rad = angles[north_idx]
        north_angle_deg_raw = np.degrees(north_angle_rad)
        
        # Apply calibration offset to correct for triangle shape bias
        north_angle_deg = (north_angle_deg_raw + NORTH_CALIBRATION_OFFSET) % 360
        
        if self.debug or True:  # Force debug for testing
            max_white_count = smoothed[north_idx]
            raw_max = white_pixel_counts[north_idx]
            print(f"[radar] North indicator detected at {north_angle_deg:.1f}° (raw: {north_angle_deg_raw:.1f}°, offset: +{NORTH_CALIBRATION_OFFSET}°)")
            
            # Save debug visualization
            debug_img = cv2.cvtColor(gray_frame.copy(), cv2.COLOR_GRAY2BGR)
            
            # Draw the radar circle boundary (blue)
            cv2.circle(debug_img, (center_x, center_y), radar_radius, (255, 0, 0), 1)
            
            # Draw the sampling ring boundaries (yellow)
            cv2.circle(debug_img, (center_x, center_y), inner_sample_radius, (0, 255, 255), 1)
            cv2.circle(debug_img, (center_x, center_y), outer_sample_radius, (0, 255, 255), 1)
            
            # Highlight detected white pixels in red (for debugging)
            for angle in angles:
                for r in range(inner_sample_radius, outer_sample_radius + 1):
                    x = int(center_x + r * np.sin(angle))
                    y = int(center_y - r * np.cos(angle))
                    if 0 <= x < w and 0 <= y < h:
                        if gray_frame[y, x] >= WHITE_THRESHOLD:
                            debug_img[y, x] = (0, 0, 255)  # Red for white pixels
            
            # Draw detected north direction (green line and dot) using calibrated angle
            north_angle_calibrated_rad = np.radians(north_angle_deg)
            north_x = int(center_x + (radar_radius + 15) * np.sin(north_angle_calibrated_rad))
            north_y = int(center_y - (radar_radius + 15) * np.cos(north_angle_calibrated_rad))
            cv2.line(debug_img, (center_x, center_y), (north_x, north_y), (0, 255, 0), 2)
            cv2.circle(debug_img, (north_x, north_y), 5, (0, 255, 0), -1)
            
            # Add text
            cv2.putText(debug_img, f"North: {north_angle_deg:.1f} deg", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(debug_img, f"White pixels: {max_white_count:.0f}", (10, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            cv2.imwrite("debug_north_detection.png", debug_img)
            print(f"[radar] Saved debug_north_detection.png")
        
        return north_angle_deg

    def _rotate_image(self, img: np.ndarray, angle_deg: float) -> np.ndarray:
        """
        Rotate image around its center by the given angle (degrees, counter-clockwise positive).
        """
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        
        # Rotation matrix
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        
        # Rotate with black border fill
        rotated = cv2.warpAffine(img, M, (w, h), borderValue=0)
        return rotated

    def _normalize_radar_rotation(self, gray_frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect the North indicator and rotate the radar so North is always at the top.
        Returns the rotated frame and the rotation angle applied.
        """
        # Detect where North currently is
        north_angle = self._detect_north_angle(gray_frame)
        
        # Rotate so North (the detected angle) moves to the top (0°)
        # OpenCV's getRotationMatrix2D rotates counter-clockwise for positive angles.
        # If North is at 90° (right side), we rotate counter-clockwise by +90° 
        # to bring it to the top (right → top when rotating counter-clockwise)
        rotation_needed = north_angle
        
        rotated = self._rotate_image(gray_frame, rotation_needed)
        
        if self.debug or True:  # Force debug for testing
            print(f"[radar] North at {north_angle:.1f}°, rotating by {rotation_needed:.1f}° to normalize")
        
        return rotated, north_angle

    def _extract_radar_content(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract the circular radar content, masking out the HUD elements.
        Uses the known geometry: 250x250px radar circle centered in 300x300 capture.
        """
        h, w = frame.shape
        scale = min(w, h) / 300.0
        
        # Radar circle is 250px diameter at 300x300
        radar_radius = int(125 * scale)
        center = (w // 2, h // 2)
        
        # Create circular mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, center, radar_radius - 5, 255, -1)  # Slightly smaller to avoid edge artifacts
        
        # Apply mask
        masked = cv2.bitwise_and(frame, frame, mask=mask)
        return masked

    def find_position(self, map_name: str, normalize_rotation: bool = True) -> Optional[Tuple[float, float]]:
        """
        Capture radar and find player position (relative 0.0-1.0).
        Uses template matching since the radar (with north at top) should match 
        almost perfectly on the map at the known scale.
        
        Args:
            map_name: Name of the map to match against
            normalize_rotation: If True, detect North indicator and rotate radar
                               so North is always at top before matching
        
        Returns:
            (x, y) relative to the full map image (0.0-1.0), or None if failed.
        """
        # Ensure map is loaded
        if not self._load_map(map_name):
            return None
            
        ref_img, ref_kp, ref_des, (pad_x, pad_y), scale_factor = self._cache[map_name]
        
        # 1. Grab current radar
        frame = self.grab_radar()
        if frame is None:
            return None
        
        # 2. Normalize rotation using North indicator
        north_angle = 0.0
        if normalize_rotation:
            frame, north_angle = self._normalize_radar_rotation(frame)
            if self.debug or True:  # Force debug for testing
                cv2.imwrite("debug_radar_normalized.png", frame)
                print(f"[radar] Saved debug_radar_normalized.png (North was at {north_angle:.1f}°)")
        
        # 3. Extract just the radar content (circular mask)
        radar_content = self._extract_radar_content(frame)
        
        # 4. Scale radar to match map using calculated scale factor
        # Scale factor is calculated based on map dimensions relative to mirage calibration
        radar_h, radar_w = radar_content.shape
        scaled_size = int(radar_w * scale_factor)
        scaled_radar = cv2.resize(radar_content, (scaled_size, scaled_size), interpolation=cv2.INTER_LINEAR)
        
        if self.debug or True:
            cv2.imwrite("debug_radar_scaled.png", scaled_radar)
            print(f"[radar] Scaled radar from {radar_w}x{radar_h} to {scaled_size}x{scaled_size} (scale: {scale_factor:.3f})")
        
        # 5. Template matching at the known scale
        ref_h, ref_w = ref_img.shape
        
        # Ensure template is not larger than reference
        if scaled_radar.shape[0] > ref_h or scaled_radar.shape[1] > ref_w:
            print(f"[radar] Scaled radar ({scaled_size}x{scaled_size}) is larger than map ({ref_w}x{ref_h})")
            return None
        
        # Template matching
        result = cv2.matchTemplate(ref_img, scaled_radar, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val < 0.2:  # Threshold for minimum match quality
            print(f"[radar] Template matching failed (score: {max_val:.3f})")
            return None
        
        # 6. Calculate player position
        # The player is at the center of the radar, which maps to the center of the matched region
        match_center_x = max_loc[0] + scaled_radar.shape[1] // 2
        match_center_y = max_loc[1] + scaled_radar.shape[0] // 2
        
        px, py = float(match_center_x), float(match_center_y)
        
        # --- DEBUG VISUALIZATION ---
        if self.debug or True:  # Force debug for testing
            debug_img = cv2.cvtColor(ref_img.copy(), cv2.COLOR_GRAY2BGR)
            
            # Draw matched region rectangle (cyan)
            top_left = max_loc
            bottom_right = (max_loc[0] + scaled_radar.shape[1], max_loc[1] + scaled_radar.shape[0])
            cv2.rectangle(debug_img, top_left, bottom_right, (255, 255, 0), 2)
            
            # Draw player position (red dot)
            cv2.circle(debug_img, (int(px), int(py)), 15, (0, 0, 255), -1)
            
            # Draw padding boundary (green rectangle)
            padded_h, padded_w = ref_img.shape
            orig_w = padded_w - 2 * pad_x
            orig_h = padded_h - 2 * pad_y
            cv2.rectangle(debug_img, (pad_x, pad_y), (pad_x + orig_w, pad_y + orig_h), (0, 255, 0), 2)
            
            # Add info text
            cv2.putText(debug_img, f"North: {north_angle:.1f} deg", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(debug_img, f"Match: {max_val:.3f} @ scale {scale_factor:.3f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imwrite("debug_player_pos.png", debug_img)
            print(f"[radar] Saved debug_player_pos.png (match score: {max_val:.3f})")
        # ---------------------------

        # 7. Convert to relative coordinates (accounting for padding)
        padded_h, padded_w = ref_img.shape
        orig_w = padded_w - 2 * pad_x
        orig_h = padded_h - 2 * pad_y
        
        # Position relative to original (unpadded) map
        orig_px = px - pad_x
        orig_py = py - pad_y
        
        rel_x = max(0.0, min(1.0, orig_px / orig_w))
        rel_y = max(0.0, min(1.0, orig_py / orig_h))
        
        if self.debug or True:
            print(f"[radar] Position: ({rel_x:.3f}, {rel_y:.3f})")
            
        return rel_x, rel_y
