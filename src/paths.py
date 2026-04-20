"""Repository root (parent of `src/`) for stable paths regardless of notebook CWD."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
