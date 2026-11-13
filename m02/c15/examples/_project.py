import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2] / "project-chat"
sys.path.insert(0, str(PROJECT / "solution"))
sys.path.insert(0, str(PROJECT / "tests"))
