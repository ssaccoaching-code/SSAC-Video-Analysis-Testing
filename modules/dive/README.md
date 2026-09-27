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

(Write down what works, what does not, and ideas for later.)
