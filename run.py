"""Run one, several or all modules from the terminal.

    python run.py --list
    python run.py dive samples/synthetic_sprint.mp4
    python run.py dive,turn clip.mov --start 1.2 --course LCM
    python run.py all clip.mov --json results/clip.json   (whole lap: every phase in order)
    python run.py event_above race.mov --start 0.8        (whole event, any number of lengths)

To run only your own module you can also use:  python modules/dive/cli.py clip.mov
"""

from __future__ import annotations

import argparse
import sys

from core.cli import add_common_arguments, options_from_args, run_modules
from modules import MODULES, get_module, module_names


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SSAC SplitSecond swim video analyzer")
    parser.add_argument("modules", nargs="?", help="module name, comma list, or 'all'")
    add_common_arguments(parser)
    parser.add_argument("--list", action="store_true", help="show the registered modules and exit")
    args = parser.parse_args(argv)

    if args.list:
        for name, cls in MODULES.items():
            owner = f" (owner: {cls.owner})" if cls.owner else ""
            print(f"{name:12} {cls.description}{owner}")
        return 0
    if not args.modules or not args.clip:
        parser.error("need a module name and a clip path (or --list)")

    if args.modules in ("all", "lap"):
        from core.cli import run_lap

        run_lap(args.clip, options_from_args(args), args.json)
        return 0
    names = [name.strip() for name in args.modules.split(",")]
    run_modules([get_module(name) for name in names], args.clip, options_from_args(args), args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
