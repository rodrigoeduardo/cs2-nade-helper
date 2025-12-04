#!/usr/bin/env python
"""
Entry point script to run the CS2 nade helper application.
"""

import sys
from src.main import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

