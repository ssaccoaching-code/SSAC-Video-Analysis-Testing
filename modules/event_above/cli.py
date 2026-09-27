"""Terminal script for the event_above module only.

    python modules/event_above/cli.py samples/synthetic_100.mp4
    python modules/event_above/cli.py clip.mov --start 1.2 --course LCM --json results/event_above.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.cli import single_module_main  # noqa: E402
from modules.event_above.module import EventAboveModule  # noqa: E402

if __name__ == "__main__":
    sys.exit(single_module_main(EventAboveModule()))
