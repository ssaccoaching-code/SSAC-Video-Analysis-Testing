"""Terminal script for the event_underwater module only.

    python modules/event_underwater/cli.py samples/synthetic_100.mp4
    python modules/event_underwater/cli.py clip.mov --start 1.2 --course LCM --json results/event_underwater.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.cli import single_module_main  # noqa: E402
from modules.event_underwater.module import EventUnderwaterModule  # noqa: E402

if __name__ == "__main__":
    sys.exit(single_module_main(EventUnderwaterModule()))
