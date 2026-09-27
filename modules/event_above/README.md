# event_above module

Owner: (your name)

## What it measures

See the docstring at the top of `module.py`. This module takes a clip of a
whole swim (any number of lengths), finds the lengths itself and runs the
dive, underwater, swim and turn modules on each one. It does not do its own
image analysis; it lines up what the four phase modules find, length by length.

## Run it

1. In Terminal, go to the project root folder (the one with `run.py`).
2. Activate the venv: `source .venv/bin/activate` (Windows: `.venv\Scripts\activate`).
3. Make a test clip with four lengths: `python tools/make_sample_clip.py samples/synthetic_100.mp4 --lengths 4 --seconds 40`
4. Terminal: `python modules/event_above/cli.py samples/synthetic_100.mp4 --start 1.0`
5. Web page: `streamlit run modules/event_above/app.py`
6. Tests: `python -m pytest modules/event_above`

## Files

| File | What it is |
|------|------------|
| `module.py` | Picks the metrics out of `core/event.py`. This is the only file the team app uses. |
| `cli.py` | Runs the module from the terminal |
| `app.py` | Runs the module as a web page |
| `helpers/` | Any extra code you write |
| `tests/` | Your tests |
| `requirements.txt` | Extra packages only you need |

## Notes

(Write down what works, what does not, and ideas for later.)
