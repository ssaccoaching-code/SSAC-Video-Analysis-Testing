# dive module

Owner: (your name)

## What it measures

See the docstring at the top of `module.py`.

## Run it

1. In Terminal, go to the project root folder (the one with `run.py`).
2. Activate the venv: `source .venv/bin/activate` (Windows: `.venv\Scripts\activate`).
3. Terminal: `python modules/dive/cli.py samples/synthetic_sprint.mp4`
4. Web page: `streamlit run modules/dive/app.py`
5. Tests: `python -m pytest modules/dive`

## Files

| File | What it is |
|------|------------|
| `module.py` | The analysis. This is the only file the team app uses. |
| `cli.py` | Runs the module from the terminal |
| `app.py` | Runs the module as a web page |
| `helpers/` | Any extra code you write |
| `tests/` | Your tests |
| `requirements.txt` | Extra packages only you need |

## Notes

- Oct 2026: entry distance used to be measured from the edge of the picture with an 8 m frame
  guess (Anthony came out at 5.88 m). Now 0 m is the block edge where the feet leave and the
  scale comes from the stretched body in flight (`swimmer_height_m`, default 1.75). Entry is
  capped at the first sustained splash after take-off, and a vertical blob (someone walking on
  deck in a clip that ends before the dive) is no longer reported as a dive.
- Known rough spots: when the swimmer overlaps people or railings behind the block at take-off
  the block edge is back-projected from the flight (a note says so); backstroke starts are not
  dives off a block, so those numbers mean little.
