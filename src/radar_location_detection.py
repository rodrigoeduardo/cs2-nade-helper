"""
Radar capture logic to identify player position on the map using Template Matching.
"""

import cv2
import numpy as np
import os
from typing import Optional, Tuple
from mss import mss

from .config import CAPTURE_MONITOR_INDEX, RADAR_REGION, DEBUG_MODE, DEBUG_SAVE_RADAR_PATH, DEBUG_SAVE_RADAR_NORMALIZED_PATH, DEBUG_SAVE_NORTH_DETECTION_PATH, DEBUG_SAVE_RADAR_SCALED_PATH, DEBUG_SAVE_PLAYER_POS_PATH 

class RadarMatcher:
    """
    Matches the current in-game radar view against a full map overview
    to determine the player's position.
    """
    
    def __init__(self, maps_dir: str = "maps") -> None:
        self.maps_dir = maps_dir
        self.sct = mss()
        
        # Cache for loaded maps: {map_name: (image, padding, scale_factor)}
        # padding = (pad_x, pad_y) added to match game radar black borders at edges
        # scale_factor = calculated based on map dimensions relative to mirage calibration
        self._cache = {}
        
        # Padding: fixed 800px on each side to handle edge cases
        # When player is at map borders, radar shows black areas outside the map
        self.padding_px = 800
        
        # Calibration constants (from mirage map)
        # Mirage: 1374x1196px, scale factor 3.57 works perfectly
        self.MIRAGE_AVG_DIMENSION = (1374 + 1196) / 2.0
        self.MIRAGE_SCALE_FACTOR = 3.57

    def _load_map(self, map_name: str) -> bool:
        """Load map reference image and add black padding for template matching."""
        if map_name in self._cache:
            return True
            
        filename = f"{map_name}.png"
        path = os.path.join(self.maps_dir, filename)
        
        if not os.path.exists(path):
            filename_legacy = f"{map_name}_radar.png"
            path_legacy = os.path.join(self.maps_dir, filename_legacy)
            if os.path.exists(path_legacy):
                path = path_legacy
            else:
                print(f"[radar] Map file not found: {path}")
                return False
            
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"[radar] Failed to load image: {path}")
            return False
        
        # Add fixed black padding to handle edge cases
        # When player is at map borders, radar shows black areas outside the map
        h, w = img.shape
        pad_x = self.padding_px
        pad_y = self.padding_px
        
        padded_img = cv2.copyMakeBorder(
            img, 
            top=pad_y, bottom=pad_y, 
            left=pad_x, right=pad_x, 
            borderType=cv2.BORDER_CONSTANT, 
            value=0
        )
        
        # Calculate scale factor based on map dimensions relative to mirage calibration
        map_avg_dimension = (w + h) / 2.0
        scale_factor = self.MIRAGE_SCALE_FACTOR * (map_avg_dimension / self.MIRAGE_AVG_DIMENSION)
        
        self._cache[map_name] = (padded_img, (pad_x, pad_y), scale_factor)
        print(f"[radar] Loaded {map_name} ({w}x{h}) with {pad_x}px padding, scale factor: {scale_factor:.3f}")
        return True

    def grab_radar(self) -> np.ndarray | None:
        """Capture the radar region."""
        region = RADAR_REGION.as_mss_dict()
        region["mon"] = CAPTURE_MONITOR_INDEX
        try:
            shot = self.sct.grab(region)
            img = np.array(shot)
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            if DEBUG_MODE:
                cv2.imwrite(DEBUG_SAVE_RADAR_PATH, gray)
                print(f"[capture] Saved debug frames to {DEBUG_SAVE_RADAR_PATH}")
            return gray
        except Exception as exc:
            print(f"[radar] Failed to grab radar: {exc}")
            return None

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
        
        scale = min(w, h) / 300.0
        radar_radius = int(125 * scale)
        inner_sample_radius = int(radar_radius + 8 * scale)
        outer_sample_radius = int(radar_radius + 20 * scale)
        
        WHITE_THRESHOLD = 250
        NORTH_CALIBRATION_OFFSET = 7.0
        
        num_samples = 360
        angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
        
        white_pixel_counts = []
        for angle in angles:
            white_count = 0
            for r in range(inner_sample_radius, outer_sample_radius + 1):
                x = int(center_x + r * np.sin(angle))
                y = int(center_y - r * np.cos(angle))
                
                if 0 <= x < w and 0 <= y < h:
                    if gray_frame[y, x] >= WHITE_THRESHOLD:
                        white_count += 1
            
            white_pixel_counts.append(white_count)
        
        white_pixel_counts = np.array(white_pixel_counts)
        
        kernel_size = 15
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(white_pixel_counts, kernel, mode='same')
        
        north_idx = np.argmax(smoothed)
        north_angle_rad = angles[north_idx]
        north_angle_deg_raw = np.degrees(north_angle_rad)
        north_angle_deg = (north_angle_deg_raw + NORTH_CALIBRATION_OFFSET) % 360
        
        if DEBUG_MODE:
            max_white_count = smoothed[north_idx]
            raw_max = white_pixel_counts[north_idx]
            print(f"[radar] North indicator detected at {north_angle_deg:.1f}° (raw: {north_angle_deg_raw:.1f}°, offset: +{NORTH_CALIBRATION_OFFSET}°)")
            
            debug_img = cv2.cvtColor(gray_frame.copy(), cv2.COLOR_GRAY2BGR)
            
            cv2.circle(debug_img, (center_x, center_y), radar_radius, (255, 0, 0), 1)
            cv2.circle(debug_img, (center_x, center_y), inner_sample_radius, (0, 255, 255), 1)
            cv2.circle(debug_img, (center_x, center_y), outer_sample_radius, (0, 255, 255), 1)
            
            for angle in angles:
                for r in range(inner_sample_radius, outer_sample_radius + 1):
                    x = int(center_x + r * np.sin(angle))
                    y = int(center_y - r * np.cos(angle))
                    if 0 <= x < w and 0 <= y < h:
                        if gray_frame[y, x] >= WHITE_THRESHOLD:
                            debug_img[y, x] = (0, 0, 255)
            
            north_angle_calibrated_rad = np.radians(north_angle_deg)
            north_x = int(center_x + (radar_radius + 15) * np.sin(north_angle_calibrated_rad))
            north_y = int(center_y - (radar_radius + 15) * np.cos(north_angle_calibrated_rad))
            cv2.line(debug_img, (center_x, center_y), (north_x, north_y), (0, 255, 0), 2)
            cv2.circle(debug_img, (north_x, north_y), 5, (0, 255, 0), -1)
            
            cv2.putText(debug_img, f"North: {north_angle_deg:.1f} deg", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(debug_img, f"White pixels: {max_white_count:.0f}", (10, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            cv2.imwrite(DEBUG_SAVE_NORTH_DETECTION_PATH, debug_img)
            print(f"[radar] Saved {DEBUG_SAVE_NORTH_DETECTION_PATH}")
        
        return north_angle_deg

    def _rotate_image(self, img: np.ndarray, angle_deg: float) -> np.ndarray:
        """Rotate image around its center by the given angle (degrees, counter-clockwise positive)."""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderValue=0)
        return rotated

    def _normalize_radar_rotation(self, gray_frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect the North indicator and rotate the radar so North is always at the top.
        Returns the rotated frame and the rotation angle applied.
        """
        # Detect where North currently is
        north_angle = self._detect_north_angle(gray_frame)
        
        # Rotate so North moves to the top (0°)
        rotation_needed = north_angle
        
        rotated = self._rotate_image(gray_frame, rotation_needed)
        
        if DEBUG_MODE:
            print(f"[radar] North at {north_angle:.1f}°, rotating by {rotation_needed:.1f}° to normalize")
        
        return rotated, north_angle

    def _extract_radar_content(self, frame: np.ndarray) -> np.ndarray:
        """Extract the circular radar content, masking out HUD elements."""
        h, w = frame.shape
        scale = min(w, h) / 300.0
        radar_radius = int(125 * scale)
        center = (w // 2, h // 2)
        
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, center, radar_radius - 5, 255, -1)
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
            
        ref_img, (pad_x, pad_y), scale_factor = self._cache[map_name]
        
        frame = self.grab_radar()
        if frame is None:
            return None
        
        # Normalize rotation using North indicator
        north_angle = 0.0
        if normalize_rotation:
            frame, north_angle = self._normalize_radar_rotation(frame)
            if DEBUG_MODE:
                cv2.imwrite(DEBUG_SAVE_RADAR_NORMALIZED_PATH, frame)
                print(f"[radar] Saved {DEBUG_SAVE_RADAR_NORMALIZED_PATH} (North was at {north_angle:.1f}°)")
        
        # Extract radar content (circular mask)
        radar_content = self._extract_radar_content(frame)
        
        # Scale radar to match map using calculated scale factor
        radar_h, radar_w = radar_content.shape
        scaled_size = int(radar_w * scale_factor)
        scaled_radar = cv2.resize(radar_content, (scaled_size, scaled_size), interpolation=cv2.INTER_LINEAR)
        
        if DEBUG_MODE:
            cv2.imwrite(DEBUG_SAVE_RADAR_SCALED_PATH, scaled_radar)
            print(f"[radar] Saved {DEBUG_SAVE_RADAR_SCALED_PATH} (Scaled radar from {radar_w}x{radar_h} to {scaled_size}x{scaled_size} (scale: {scale_factor:.3f})")
        
        # Template matching
        ref_h, ref_w = ref_img.shape
        
        if scaled_radar.shape[0] > ref_h or scaled_radar.shape[1] > ref_w:
            print(f"[radar] Scaled radar ({scaled_size}x{scaled_size}) is larger than map ({ref_w}x{ref_h})")
            return None
        
        result = cv2.matchTemplate(ref_img, scaled_radar, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val < 0.2:
            print(f"[radar] Template matching failed (score: {max_val:.3f})")
            return None
        
        # Calculate player position (center of matched region)
        match_center_x = max_loc[0] + scaled_radar.shape[1] // 2
        match_center_y = max_loc[1] + scaled_radar.shape[0] // 2
        
        px, py = float(match_center_x), float(match_center_y)
        
        if DEBUG_MODE:
            debug_img = cv2.cvtColor(ref_img.copy(), cv2.COLOR_GRAY2BGR)
            
            top_left = max_loc
            bottom_right = (max_loc[0] + scaled_radar.shape[1], max_loc[1] + scaled_radar.shape[0])
            cv2.rectangle(debug_img, top_left, bottom_right, (255, 255, 0), 2)
            cv2.circle(debug_img, (int(px), int(py)), 15, (0, 0, 255), -1)
            
            padded_h, padded_w = ref_img.shape
            orig_w = padded_w - 2 * pad_x
            orig_h = padded_h - 2 * pad_y
            cv2.rectangle(debug_img, (pad_x, pad_y), (pad_x + orig_w, pad_y + orig_h), (0, 255, 0), 2)
            
            cv2.putText(debug_img, f"North: {north_angle:.1f} deg", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 3.5, (255, 255, 0), 10)
            cv2.putText(debug_img, f"Match: {max_val:.3f} @ scale {scale_factor:.3f}", (10, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 3.5, (255, 255, 0), 10)

            cv2.imwrite(DEBUG_SAVE_PLAYER_POS_PATH, debug_img)
            print(f"[radar] Saved {DEBUG_SAVE_PLAYER_POS_PATH} (match score: {max_val:.3f})")

        # Convert to relative coordinates (accounting for padding)
        padded_h, padded_w = ref_img.shape
        orig_w = padded_w - 2 * pad_x
        orig_h = padded_h - 2 * pad_y
        
        orig_px = px - pad_x
        orig_py = py - pad_y
        
        rel_x = max(0.0, min(1.0, orig_px / orig_w))
        rel_y = max(0.0, min(1.0, orig_py / orig_h))
        
        if DEBUG_MODE:
            print(f"[radar] Position: ({rel_x:.3f}, {rel_y:.3f})")
            
        return rel_x, rel_y
