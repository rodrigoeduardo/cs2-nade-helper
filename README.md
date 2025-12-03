# CS2 Nades Helper

This project helps Counter-Strike 2 players by detecting the currently equipped grenade and providing relevant lineup information.

ATTENTION! THIS PROJECT IS 90% VIBECODED!

## Purpose

The CS2 Nades Helper runs in the background and:

1.  **Captures the screen** in the weapon HUD area.
2.  **Uses OCR (Optical Character Recognition)** to detect which grenade (Smoke, Molotov, Flashbang, HE) is currently equipped.
3.  **Displays desktop notifications** with the detected grenade.
4.  **(Optional) Opens a browser window** showing lineups for the detected grenade on the current map (currently defaults to Mirage).

## Prerequisites

- **Python 3.8+**
- **Tesseract OCR**: This project requires Tesseract OCR to be installed on your system.
  - Download and install from [UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki).
  - By default, the project looks for Tesseract at `C:\Program Files\Tesseract-OCR\tesseract.exe`. If you install it elsewhere, update `TESSERACT_CMD` in `config.py`.

## Installation

1.  Clone this repository:

    ```bash
    git clone <repository-url>
    cd cs2-nades-helper
    ```

2.  Create a virtual environment (recommended):

    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # On Windows PowerShell
    ```

3.  Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

Open `config.py` to adjust settings:

- **HUD_REGION**: You may need to adjust the `top`, `left`, `width`, and `height` values to match your screen resolution and HUD scale. The defaults are for 1920x1080.
- **TESSERACT_CMD**: Path to your Tesseract executable.

## Usage

Run the helper script:

```bash
python main.py
```

### Enable Browser Lineups

To automatically open a browser with lineups for the detected grenade:

```bash
python main.py --browser
```

Press `Ctrl+C` in the terminal to stop the application.
