<!-- e9dece05-586e-4e4a-b1c4-f4533618dabb 5f50a53f-c996-419f-b572-4e75a7d26f71 -->
# CS2 Nade Helper Plan

1. **Architecture & Setup**

- Define a modular entry point in [`main.py`](main.py) that orchestrates capture, classification, and browser control loops.
- Configure a settings module (e.g., `config.py`) for map list, screen-region coordinates, refresh intervals, and URLs pulled from [csnades.gg](https://csnades.gg/).

2. **Screen Capture Pipeline**

- Use `mss` for high-frequency grabs of two regions: the weapon HUD (bottom-left) and the radar/minimap.
- Normalize captures (resize, grayscale, denoise) with `opencv-python` to improve OCR/classifier accuracy.

3. **Nade Detection**

- For textual HUD labels, integrate `pytesseract` (with tuned whitelist) to read the held grenade name.
- Back up OCR with template matching or a lightweight classifier (`opencv` feature matching or `onnxruntime` with a tiny CNN) to disambiguate smokes vs molotovs when text confidence is low.

4. **Map/Position Detection**

- Assume the CS2 radar is static (no rotation); store per-map calibration data (radar top-left pixel, width/height, and sector bounding boxes) in `config.py` or a JSON file.
- Use color-thresholding to isolate the player triangle, convert to normalized radar coordinates, and map the location into predefined sectors/callouts via lookup tables.

5. **Browser Automation**

- Automate a Chromium instance via `playwright` (preferred for async control) or `selenium` if Playwright is unavailable.
- Build URL routing logic: given map + sector + grenade type, open the matching csnades.gg page (fallback to general map guide if no exact entry).
- Cache opened tabs and focus/refresh instead of launching duplicates.

6. **Controller Loop & UI Feedback**

- Run an async loop that polls capture tasks, debounces changes, and triggers navigation only when both nade type and sector are confidently detected.
- Provide a lightweight overlay or desktop notification (e.g., `pystray` or `plyer`) showing the detected state and the URL opened for transparency.

7. **Testing & Calibration**

- Add diagnostic scripts to log OCR confidence and radar coordinates for manual calibration per resolution.
- Include configuration presets for common CS2 resolutions/aspect ratios so users can tune without editing code.

### To-dos

- [ ] Define modules config, capture, detection, browser
- [ ] Implement MSS capture + preprocessing
- [ ] Add OCR/template logic for held grenade
- [ ] Map radar position to callouts
- [ ] Automate csnades.gg navigation
- [ ] Controller loop + overlay feedback