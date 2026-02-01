#!/usr/bin/env python3
"""
Backward Compatibility Wrapper for analyze_experiments.py

This script forwards old-style Priority 1 analysis commands to the new
generalized analyze_experiments.py script.
"""

import sys
import subprocess
from pathlib import Path

# Get the directory containing this script
script_dir = Path(__file__).parent
analyze_experiments = script_dir / "analyze_experiments.py"

if __name__ == "__main__":
    # Forward all arguments to analyze_experiments.py with priority=1 prepended
    cmd = [sys.executable, str(analyze_experiments), "1"] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))
