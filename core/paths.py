"""Lets a script inside modules/<name>/ import core and modules.

Put these two lines at the top of any script that lives in a module folder:

    from pathlib import Path; import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
RESULTS = ROOT / "results"
