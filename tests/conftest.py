# tests/conftest.py
# Shared configuration for pytest

import sys
import pathlib

# Make sure project root is on the path
ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "webapp"))
