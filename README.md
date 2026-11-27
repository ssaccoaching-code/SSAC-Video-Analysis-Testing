# SSAC SplitSecond

Swim race video analyzer for the SSAC team.

- Input: a video of one swimmer's race.
- Output: numbers for each race phase (dive, underwater, swim, turn).
- Built as 4 separate modules so 4 people can work at the same time.
- Every module can be run two ways: from the terminal, or as a web page.

## What is where

| Path | What it is | Who edits it |
|------|------------|--------------|
| `core/` | Shared code: video reading, swimmer tracking (`track.py`), calibration (`calibrate.py`), speed series (`analysis.py`), whole-lap runner (`lap.py`), whole-event runner (`event.py`), summary and improvement plan (`coach.py`), reports (`render.py`), terminal and web page helpers | Elaine and Ken |
| `modules/dive/` | Dive module: reaction, flight, entry | Theo |
| `modules/underwater/` | Underwater module: kicks, breakout | (name) |
| `modules/swim/` | Swim module: strokes, tempo, splits | (name) |
| `modules/turn/` | Turn module: flip and open turns | (name) |
| `modules/event_above/` | Whole event from an above-water camera, any number of lengths (runs the four phase modules on every length) | (name) |
| `modules/event_underwater/` | Whole event from an underwater camera, any number of lengths | (name) |
| `modules/__init__.py` | The list of modules (`MODULES`) | Theo |
| `run.py` | Team terminal script: runs any or all modules | Theo |
| `app.py` | Team web page: pick a phase or the whole lap, upload, get the report | Theo |
| `calibration/` | Saved camera calibrations (JSON), written by the web page | anyone |
| `tests/` | Tests for `core/` and the shared contract test | Elaine and Ken |
| `tools/make_sample_clip.py` | Makes a cartoon clip for testing | anyone |
| `samples/` | Put video clips here (not saved to git) | anyone |
| `results/` | JSON output goes here (not saved to git) | anyone |
| `CODEOWNERS` | Who reviews which folder on GitHub | SSAC |

## Inside each module folder

| File | What it is |
|------|------------|
| `module.py` | The analysis. The only file the team app uses. Must follow the contract in `core/base.py`. |
| `cli.py` | Runs this module from the terminal |
| `app.py` | Runs this module as a web page |
| `helpers/` | Any extra code the owner writes |
| `tests/` | The owner's tests |
| `requirements.txt` | Extra packages only this module needs |
| `README.md` | The owner's notes |

Rule: you edit only your own module folder. Nothing else.

## First time setup

Mac:

1. Open Terminal.
2. `cd` into this folder.
3. Run `./setup.sh`.
4. Wait until it says `Done`.

Windows:

1. Double-click `setup.bat`.
2. Wait until it says `Done`.

This creates a `.venv` folder with OpenCV, numpy, Streamlit, matplotlib and pytest inside.
If you set up before matplotlib was added, run `pip install matplotlib` once with the venv active.

## Every time you open a new terminal

1. `cd` into this folder (the one with `run.py`).
2. Mac: `source .venv/bin/activate`. Windows: `.venv\Scripts\activate`.
3. You should see `(.venv)` at the start of the prompt.

## Get a test clip

1. Run `python tools/make_sample_clip.py`.
2. It writes `samples/synthetic_sprint.mp4`, a cartoon of one length.
3. For a whole event: `python tools/make_sample_clip.py samples/synthetic_100.mp4 --lengths 4 --seconds 40`.

## Run your module from the terminal

1. `python modules/dive/cli.py samples/synthetic_sprint.mp4` (swap `dive` for yours).
2. The numbers print in the terminal.
3. To save them: add `--json results/dive.json`.

Options:

- `--start 1.2` : the start signal is at 1.2 seconds in the clip
- `--course SCY` : pool type. `SCY` (25 yd, default), `SCM` (25 m) or `LCM` (50 m)
- `--lane 5` : the swimmer's lane
- `-o key=value` : any extra setting your module needs

## Run your module as a web page

1. `streamlit run modules/dive/app.py` (swap `dive` for yours).
2. A browser tab opens.
3. Set the pool and start time in the left sidebar.
4. Click Upload and pick a clip.
5. Click Analyze.
6. Read the numbers. Key frames show as pictures. Download JSON if you want the file.
7. To stop: go back to the terminal and press Ctrl+C.

## Run everything (the team app)

Web page (what coaches use):

1. `streamlit run app.py`
2. Pick what is in the video: Dive, Underwater / breakout, Swim, Turn, Whole lap above water, Whole lap under water,
   Whole event above water, or Whole event under water. The two "whole event" options take a clip of a complete swim
   with any number of lengths (a 50, a 100, a 200 ...): the app finds the lengths itself and runs every phase on each one.
3. Upload the clip and click Analyze. A progress bar runs while the swimmer is found.
   If the clip does not contain what you picked (a turn, say, in a video with no wall), the page says so
   and asks for the right kind of video instead of showing numbers.
4. Read the summary and the tentative improvement plan, then the numbers, the key frames and the speed chart.
   For a whole event the numbers are by length (splits, speed, strokes, tempo, underwater distance, turn times)
   and the per-length details sit in an expander underneath.
5. Download the report card (PNG), the report (PDF), the numbers (JSON) or the overlay video (MP4 with the live speed in the corner).
6. Type the swimmer's name and press Save these results to keep the numbers in the database.

The page has three tabs:

- Analyze: one clip, as above.
- Compare: the same phase on two to four clips (different swimmers, or one swimmer over time). Same depth
  as a single report, side by side: the speed curves on one chart (lined up on the entry, breakout or wall),
  "What differs" in plain sentences with the reference band for each number, the numbers with good / ok / poor,
  who is ahead and the gap, the key moments as frames next to each other (entry against entry ...), each clip's
  own summary, plan and analysis notes, and what each swimmer can borrow from the others. With AI review on,
  the AI compares the key frames and numbers head to head and writes a plan per swimmer. Download the
  comparison as Markdown or JSON, or save all clips to the database in one go. (core/coach.py compare(),
  core/ai.py compare_review(), core/report.py comparison_markdown()).
- Saved results: everything in the database, one run at a time, and a CSV download of all of it.

The database is a SQLite file, `results/splitsecond.db` (two tables: `runs` and `metrics`). It opens in any
SQLite viewer and is easy to move to a shared database later; see `core/db.py`.

Terminal:

- One phase: `python run.py dive clip.mov`
- Several: `python run.py dive,turn clip.mov`
- Whole lap, every phase in order: `python run.py all clip.mov --json results/clip.json`
- Whole event, any number of lengths: `python run.py event_above race.mov --start 0.8` (or `event_underwater`)
- List the modules: `python run.py --list`

Every report (terminal, JSON, PNG card, PDF, web page) ends with a summary and a tentative improvement plan.
Both are rule based: each number is compared with a reference band for club-level age-group and high-school
swimmers (the bands are in `core/coach.py`, `BANDS`, and a coach can change them) and turned into plain
sentences, a focus, a drill and a target. Nothing leaves the laptop.

## Calibration (metres per pixel)

Distances and speeds need to know how big a pixel is. The app tries, in this order:

1. A saved calibration for the camera position (sidebar, Camera calibration). Make one once per
   camera spot: tick "Mark a new calibration", read two x positions off the yellow grid
   (for example the wall and the 15 m mark, or two marks on the gutter ruler), type their
   metres from the start wall, save it under a name. Next time pick that name.
2. Both end walls in the picture (the side whole-lap camera): automatic, medium confidence.
3. `-o frame_span_m=10` on the terminal: how many metres the picture is wide.
4. A guess, marked low confidence in the notes.

The dive module does not use the guess (4). A dive camera almost never has the start wall at the
edge of the picture, so instead it puts 0 m at the block edge where the feet leave and takes
the scale from the swimmer's own stretched body in flight (about 1.28 x height). Set the
swimmer's height in the sidebar (More) or with `-o swimmer_height_m=1.80`; without it 1.75 m is
assumed. That is still a rough number: a saved calibration (1) is the real answer.

## What each camera can and cannot measure

- Side camera, whole lap (above water): strokes, tempo, speed, splits, turn time, breakout time.
  The underwater part is faint from above, so kick count is rough.
- Dive camera (above water, side on): reaction, flight, entry angle, entry distance. Depth needs the underwater camera.
- Underwater cameras: entry depth, kick count, underwater distance and time, breakout speed.
  Cut the long recordings into one clip per swimmer first (a few seconds before the start to a few seconds after the breakout).
- Whole event (either camera, fixed and showing the whole lane): splits and speed by length, turn times at every wall,
  breakout distance after every wall, strokes and tempo by length (above water) or kicks by length (under water).
  The camera has to stay still and see both walls, or have a saved calibration.

## AI review (ChatGPT or Claude)

What it does, on top of the built-in rule-based advice:

1. Looks at the key frames (entry, breakout, wall) and says if each one is really that moment.
2. Judges every number against the reference bands and the swimmer's saved history.
3. Flags numbers that look like measurement errors.
4. Writes a summary, a ranked improvement plan, and a technical report (Markdown).

Set it up (once):

1. Make a file called `.env` in the project folder (next to `run.py`).
2. Put the key in it, one line: `OPENAI_API_KEY=sk-proj-...` (ChatGPT) or `ANTHROPIC_API_KEY=sk-ant-...` (Claude).
3. If both keys are there, add `AI_PROVIDER=openai` or `AI_PROVIDER=claude` to choose.
4. Optional: pick a model with `OPENAI_MODEL=gpt-6-sol` or `ANTHROPIC_MODEL=claude-sonnet-5`.
5. Never put a key in code or in git. `.env` is git-ignored already.
6. Check it works: `python tools/ai_check.py`

Use it on the web page:

1. Open the sidebar section "AI review" and tick the box.
2. Type the swimmer's name (pulls their saved history) and any comments.
3. Press Analyze. The AI summary and plan appear above the numbers.
4. Download the report with "Technical report (Markdown)".

Use it from the terminal:

- `python run.py all clip.mov --ai --swimmer Theo --report results/theo_report.md`
- `python modules/dive/cli.py clip.mov --ai`

Cost: one request per clip, about 10 pictures plus text. A few cents with
`gpt-6-sol` or `claude-sonnet-5`. Tracking and timing still run locally and cost nothing.

## Run the tests

- Only your module: `python -m pytest modules/dive`
- Everything: `python -m pytest`
- Without pytest: `python -m unittest discover -s modules/dive/tests -t .`
- Mac shortcut: double-click `Run Tests.command`

Every module gets 6 contract tests for free. They check that the module:

1. Has a name and metric keys.
2. Is in the `MODULES` list.
3. Returns a `Result` with every metric key.
4. Only reports key frames that exist in the clip.
5. Can be saved as JSON.
6. Works for SCY, SCM and LCM.

These must stay green. Add your own tests under them.

## Build your module

1. Open `modules/<yours>/module.py`.
2. Put your name in `owner = ""`.
3. Read the docstring at the top. It lists your metrics from the team meeting.
   They are the `metric_keys`. Keep them. Add more if you need to.
4. Write the `analyze(self, clip, options)` method:
   1. Start with `result = self.empty_result()`.
   2. Loop over frames: `for index, t, frame in clip.frames(start_s=options["start_s"]):`
   3. Fill in `result.metrics["..."] = number`.
   4. Fill in `result.key_frames["entry"] = index` for important moments.
   5. Add warnings with `result.notes.append("...")`.
   6. `return result`.
5. Run your tests: `python -m pytest modules/<yours>`
6. Run it: `python modules/<yours>/cli.py samples/synthetic_sprint.mp4`
   or `streamlit run modules/<yours>/app.py`
7. Repeat 4 to 6 until the numbers look right.
8. Need a package that is not installed? Add it to `modules/<yours>/requirements.txt`
   and run `pip install -r modules/<yours>/requirements.txt`.

Shared analysis you can build on (see the docstrings):

- `core.analysis.prepare(clip, options, phase=...)` : tracks the swimmer once, calibrates, and gives
  `x_m` (metres from the start wall per frame), `speed` (m/s), the `track` (boxes, areas, foam count)
- `core.signal` : moving averages, peak and rhythm counting, crossing times
- `core.calibrate` : pixels to metres, saved profiles, water mask, wall detection

Things you can use inside `analyze`:

- `frame` : a numpy array, height x width x 3, BGR colors (OpenCV style)
- `clip.fps`, `clip.width`, `clip.height`, `clip.frame_count`, `clip.duration_s`
- `clip.time_to_frame(seconds)` and `clip.frame_to_time(index)`
- `clip.frame_at(index)` : one frame
- `options["pool_length_m"]`, `options["course"]`, `options["start_s"]`,
  `options["lane"]`, `options["swimmer_box"]`

## Using another language

If you want to write your analysis in something other than Python:

1. Put your program in `modules/<yours>/helpers/`.
2. In `module.py`, make `analyze` start your program with `subprocess.run`,
   pass it the clip path and the options as JSON, and read the JSON it prints.
3. Turn that JSON into the `Result`. The team app never knows the difference.

## Rules

1. Edit only your own module folder.
2. Do not put video files in git. `samples/` and `results/` are ignored already.
3. Do not edit anything in `core/` on your own. Ask the team first.
4. Never crash the whole run. If you cannot measure something, leave it `None`
   and explain in `result.notes`.
5. Do not hard-code the pool length. Read `options["pool_length_m"]`.
6. Units: metres, seconds, degrees, strokes per minute.
7. Code two modules both need goes in `core/` after a quick chat.

## Push to GitHub (Elaine and Ken)

1. Double-click `Push to GitHub.command`.
2. If it says the repo does not exist yet:
   1. Sign in to GitHub as theoq71.
   2. Create an empty public repo called `ssac-splitsecond` (no README).
   3. Double-click `Push to GitHub.command` again.
3. Check https://github.com/theoq71/ssac-splitsecond
