"""Web page for the event_above module only.

    streamlit run modules/event_above/app.py

Upload a clip, press Analyze, read the numbers. Everything on the page comes
from core/ui.py, so this file stays tiny. If you want extra charts or images
for your module, add them here below module_page().
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.ui import module_page  # noqa: E402
from modules.event_above.module import EventAboveModule  # noqa: E402

module_page([EventAboveModule()], "event_above module")
