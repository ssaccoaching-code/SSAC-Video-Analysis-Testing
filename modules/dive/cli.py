"""Terminal script for the dive module only.

    python modules/dive/cli.py samples/synthetic_sprint.mp4
    python modules/dive/cli.py clip.mov --start 1.2 --course LCM --json results/dive.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.cli import single_module_main  # noqa: E402
from modules.dive.module import DiveModule  # noqa: E402

if __name__ == "__main__":
    sys.exit(single_module_main(DiveModule()))
