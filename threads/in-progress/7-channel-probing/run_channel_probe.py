#!/usr/bin/env python3
"""Thin CLI wrapper for Phase 1 channel probing."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpt_oss_interp.steering.probing import main


if __name__ == "__main__":
    raise SystemExit(main())
