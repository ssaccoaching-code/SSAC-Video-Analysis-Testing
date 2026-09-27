"""The list of analysis modules.

Adding a module: create a folder here (copy an existing one), then add one
line to MODULES below. run.py, app.py and the tests find it through this dict.
"""

from core.base import AnalysisModule
from modules.dive import DiveModule
from modules.event_above import EventAboveModule
from modules.event_underwater import EventUnderwaterModule
from modules.swim import SwimModule
from modules.turn import TurnModule
from modules.underwater import UnderwaterModule

MODULES: dict[str, type[AnalysisModule]] = {
    DiveModule.name: DiveModule,
    UnderwaterModule.name: UnderwaterModule,
    SwimModule.name: SwimModule,
    TurnModule.name: TurnModule,
    EventAboveModule.name: EventAboveModule,
    EventUnderwaterModule.name: EventUnderwaterModule,
}


def get_module(name: str) -> AnalysisModule:
    try:
        return MODULES[name]()
    except KeyError:
        raise KeyError(f"no module called '{name}'; choose from {', '.join(MODULES)}") from None


def module_names() -> list[str]:
    return list(MODULES)
