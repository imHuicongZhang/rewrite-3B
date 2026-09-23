#!/usr/bin/env python
"""Run the whole suite with stdlib unittest (no pytest needed).

    <env>/bin/python tests/run_all.py [-v]
"""
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

if __name__ == "__main__":
    verbosity = 2 if "-v" in sys.argv else 1
    suite = unittest.TestLoader().discover(str(HERE), pattern="test_*.py")
    res = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    sys.exit(0 if res.wasSuccessful() else 1)
