"""Shared Streamlit page pieces.

app.py (the team page) and every modules/<name>/app.py (one module) use this,
so a module builder does not have to learn Streamlit to get a working page.

Run a page with:  streamlit run app.py
"""

from __future__ import annotations

import io
import os
import tempfile
from typing import Any

import streamlit as st

from core.base import AnalysisModule
from core.calibrate import Calibration, background_image, list_profiles, save_profile
from core.paths import RESULTS
from core.report import LABELS, coverage_message, format_value, to_json
from core.types import Result
from core.video import Clip, open_clip

# label on the page -> (module or "lap", camera view it expects or None)
PHASES = {
    "Dive": ("dive", None),
    "Underwater / breakout": ("underwater", None),
    "Swim": ("swim", None),
    "Turn": ("turn", None),
    "Whole lap above water": ("lap", "above"),
    "Whole lap under water": ("lap", "underwater"),
    "Whole event above water": ("event_above", "above"),
    "Whole event under water": ("event_underwater", "underwater"),
}


def sidebar_options() -> dict[str, Any]:
    """The same settings the command line takes, as sidebar widgets."""
    st.sidebar.header("Settings")
    course = st.sidebar.selectbox("Pool", ["SCY", "SCM", "LCM"], index=0, help="SCY = 25 yards, SCM = 25 m, LCM = 50 m")
    start_s = st.sidebar.number_input("Start signal at (seconds in clip, 0 = unknown)", min_value=0.0, value=0.0, step=0.1)
    options: dict[str, Any] = {"course": course, "start_s": float(start_s)}
    with st.sidebar.expander("Camera calibration", expanded=False):
        st.caption("Once per camera position. Tells the app how many metres a pixel is.")
        profiles = list_profiles()
        choice = st.selectbox("Saved calibration", ["Automatic"] + profiles, index=0)
        if choice != "Automatic":
            options["calib_profile"] = choice
        st.session_state["calib_new"] = st.checkbox("Mark a new calibration on the uploaded clip")
    with st.sidebar.expander("More", expanded=False):
        view = st.selectbox("Camera view", ["auto", "above", "underwater"], index=0)
        if view != "auto":
            options["view"] = view
        turn_type = st.selectbox("Turn type", ["auto", "flip", "open"], index=0)
        if turn_type != "auto":
            options["turn_type"] = turn_type
        make_video = st.checkbox("Also make the overlay video (slower)", value=True)
        st.session_state["make_video"] = make_video
    with st.sidebar.expander("AI review", expanded=False):
        from core import ai

        if ai.available():
            use_ai = st.checkbox(f"Ask {ai.provider_name()} to review the numbers and key frames", value=True)
            model = st.selectbox("Model", ai.MODELS, index=ai.MODELS.index(ai.model_name()) if ai.model_name() in ai.MODELS else 0)
            st.caption("Checks the key frames, judges the numbers against the bands and history, writes the report.")
        else:
            use_ai = False
            model = ai.DEFAULT_MODEL
            st.caption("No API key found. Put OPENAI_API_KEY=... or ANTHROPIC_API_KEY=... in a file called .env in the project folder and restart.")
        st.session_state["use_ai"] = use_ai
        st.session_state["ai_model"] = model
    return options


def upload_clip() -> str | None:
    """Show an upload box. Returns a path to the saved file, or None."""
    uploaded = st.file_uploader("Upload a race clip", type=["mp4", "mov", "m4v", "avi"])
    if uploaded is None:
        return None
    key = f"{uploaded.name}-{uploaded.size}"
    if st.session_state.get("upload_key") != key:
        suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
        handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        handle.write(uploaded.getbuffer())
        handle.close()
        st.session_state["upload_key"] = key
        st.session_state["upload_path"] = handle.name
        st.session_state["upload_name"] = uploaded.name
        st.session_state.pop("results", None)
    return st.session_state["upload_path"]


def calibration_panel(path: str, options: dict[str, Any]) -> None:
    """Mark two reference points on a frame of the clip and save them as a profile."""
    if not st.session_state.get("calib_new"):
        return
    st.subheader("Calibrate this camera")
    st.caption("Pick two things in the picture whose distance you know (the wall and the flags, two marks on the gutter ruler, "
               "the wall and the 15 m mark). Read their x position off the grid and type it in with the metres from the start wall.")
    with open_clip(path) as clip:
        frame_t = st.slider("Frame to look at (seconds)", 0.0, max(0.1, clip.duration_s), min(1.0, clip.duration_s), 0.1)
        frame = clip.frame_at(clip.time_to_frame(frame_t)).copy()
    import cv2

    step = 100 if frame.shape[1] <= 2000 else 200
    for x in range(0, frame.shape[1], step):
        cv2.line(frame, (x, 0), (x, frame.shape[0]), (0, 255, 255), 1)
        cv2.putText(frame, str(x), (x + 3, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    st.image(frame[:, :, ::-1], use_container_width=True)
    c1, c2, c3, c4 = st.columns(4)
    x1 = c1.number_input("Point 1: x (pixels)", min_value=0, value=100, step=10)
    m1 = c2.number_input("Point 1: metres from start wall", min_value=0.0, value=0.0, step=0.5)
    x2 = c3.number_input("Point 2: x (pixels)", min_value=0, value=1000, step=10)
    m2 = c4.number_input("Point 2: metres from start wall", min_value=0.0, value=5.0, step=0.5)
    name = st.text_input("Name for this camera position", value="side camera")
    if st.button("Save calibration"):
        if x1 == x2 or m1 == m2:
            st.error("The two points must differ in both pixels and metres.")
        else:
            cal = Calibration([(float(x1), float(m1)), (float(x2), float(m2))], method="points", confidence="high")
            saved = save_profile(name, cal)
            st.success(f"Saved. Pick '{name}' under Saved calibration in the sidebar. ({saved})")
    options["calib_points"] = f"{x1}:{m1},{x2}:{m2}"


def show_result(result: Result, clip: Clip | None = None) -> None:
    st.subheader(result.module.replace("_", " ").capitalize())
    rows = [{"metric": LABELS.get(key, key), "value": format_value(value)} for key, value in result.metrics.items()]
    st.table(rows)
    for note in result.notes:
        st.caption(note)
    if clip is not None and result.key_frames:
        items = list(result.key_frames.items())
        columns = st.columns(min(4, len(items)))
        for column, (label, index) in zip(columns * 10, items):
            frame = clip.frame_at(index)
            column.image(frame[:, :, ::-1], caption=f"{label} ({clip.frame_to_time(index):.2f} s)", use_container_width=True)
    if result.details:
        with st.expander(f"Details by length ({len(result.details)} results)"):
            for detail in result.details:
                st.markdown(f"**{detail.module.replace('_', ' ')}**")
                rows = [{"metric": LABELS.get(k, k), "value": format_value(v)} for k, v in detail.metrics.items() if v is not None]
                if rows:
                    st.table(rows)
                for note in detail.notes:
                    if note.startswith(f"{detail.module.split('_')[0]}:"):
                        st.caption(note)


def show_advice(results: list[Result]) -> None:
    """The summary and the tentative improvement plan, above the numbers."""
    from core.coach import coach

    advice = coach(results)
    st.subheader("Summary")
    for sentence in advice.summary:
        st.write(sentence)
    st.subheader("Improvement plan (tentative)")
    if advice.plan:
        st.table([{"focus": p["focus"], "why": p["why"], "drill": p["drill"], "target": p["target"]} for p in advice.plan])
    st.caption(advice.basis)


def run_analysis(path: str, phase_key: str, options: dict[str, Any], make_video: bool,
                 expected_view: str | None = None) -> dict[str, Any]:
    """Run the chosen phase (or the whole lap) and build every report file."""
    from core.analysis import detect_view
    from core.calibrate import background_image
    from core.lap import analyze_lap
    from core.render import overlay_video, phase_windows, report_card_png, report_pdf, speed_chart_png, speed_profile
    from modules import get_module

    bar = st.progress(0.0, text="Finding the swimmer...")
    options = dict(options)
    options["_progress"] = lambda frac: bar.progress(min(0.6, 0.6 * frac), text="Finding the swimmer...")
    out: dict[str, Any] = {}
    with open_clip(path) as clip:
        # which camera is this? If the coach said, check it; otherwise detect it
        options.setdefault("_background", {})[id(clip)] = background_image(clip)
        seen_view = detect_view(options["_background"][id(clip)])
        if expected_view and seen_view != expected_view and "view" not in options:
            out["message"] = coverage_message([], phase_key, expected_view, seen_view)
            bar.progress(1.0, text="Done")
            out.update(results=[], chart_png=None, phases=[], json="{}", card_png=None, pdf=None, video=None, key_frames={})
            return out
        if expected_view:
            options["view"] = expected_view
        if phase_key == "lap":
            results = analyze_lap(clip, options)
        else:
            results = [get_module(phase_key).run(clip, options)]
        out["message"] = coverage_message(results, phase_key)
        if out["message"]:
            bar.progress(1.0, text="Done")
            out.update(results=results, chart_png=None, phases=[], json=to_json(results, clip_path=path), card_png=None,
                       pdf=None, video=None, key_frames={})
            return out
        bar.progress(0.65, text="Drawing the report...")
        profile = speed_profile(options)
        phases = phase_windows(results, clip)
        chart = speed_chart_png(profile, phases) if profile is not None else None
        out["results"] = results
        out["chart_png"] = chart
        out["phases"] = phases
        out["json"] = to_json(results, clip_path=st.session_state.get("upload_name", path))
        title = f"SSAC SplitSecond: {st.session_state.get('upload_name', os.path.basename(path))}"
        out["card_png"] = report_card_png(results, clip, profile, title, chart)
        out["pdf"] = report_pdf(results, clip, profile, title, phases)
        out["video"] = None
        if make_video:
            bar.progress(0.75, text="Writing the overlay video...")
            stem = os.path.splitext(st.session_state.get("upload_name", "clip"))[0]
            RESULTS.mkdir(parents=True, exist_ok=True)
            video_path = str(RESULTS / f"{stem}_overlay.mp4")
            try:
                overlay_video(clip, profile, phases, video_path,
                              progress=lambda frac: bar.progress(0.75 + 0.25 * frac, text="Writing the overlay video..."))
                out["video"] = video_path
            except Exception as error:
                out["video_error"] = str(error)
        out["key_frames"] = {(r.module, label): clip.frame_at(index)[:, :, ::-1]
                             for top in results for r in [top] + list(top.details) for label, index in r.key_frames.items()}
        out["clip_info"] = str(clip)
    if st.session_state.get("use_ai"):
        from core import ai

        bar.progress(0.97, text="Asking the AI to review...")
        out["ai"] = ai.review(results, out["key_frames"], clip_info=out.get("clip_info", ""), options=options,
                              swimmer=st.session_state.get("swimmer", ""), comments=st.session_state.get("ai_comments", ""),
                              phase=phase_key, model=st.session_state.get("ai_model"))
    bar.progress(1.0, text="Done")
    return out


def show_ai_review(out: dict[str, Any]) -> None:
    """The AI summary, plan and frame checks. Falls back to the rule-based advice."""
    from core import ai

    review_out = out.get("ai")
    if review_out is None:
        show_advice(out["results"])
        return
    if not review_out.ok:
        st.warning(f"AI review not available: {review_out.error}")
        show_advice(out["results"])
        return
    st.subheader("Summary (AI review)")
    for sentence in review_out.summary:
        st.write(sentence)
    c1, c2 = st.columns(2)
    if review_out.strengths:
        c1.markdown("**Strengths**")
        for item in review_out.strengths:
            c1.write(f"- {item}")
    if review_out.weaknesses:
        c2.markdown("**To work on**")
        for item in review_out.weaknesses:
            c2.write(f"- {item}")
    st.subheader("Improvement plan")
    if review_out.plan:
        st.table([{"focus": p["focus"], "why": p["why"], "drill": p["drill"], "target": p["target"]} for p in review_out.plan])
    if review_out.frame_checks:
        with st.expander(f"Key frame checks ({len(review_out.frame_checks)})", expanded=False):
            st.table([{"module": c.get("module", ""), "frame": c.get("label", ""),
                       "looks right": "yes" if c.get("looks_right") else "no", "comment": c.get("comment", "")}
                      for c in review_out.frame_checks])
    if review_out.corrections:
        st.caption("The AI thinks these numbers are measurement errors: " +
                   ", ".join(f"{k} -> {v}" for k, v in review_out.corrections.items()))
    st.caption(review_out.basis)
    stem = os.path.splitext(st.session_state.get("upload_name", "clip"))[0]
    report = ai.report_with_numbers(review_out, out["results"], f"SSAC SplitSecond: {st.session_state.get('upload_name', stem)}")
    with st.expander("Technical report", expanded=False):
        st.markdown(review_out.report_md)
    st.download_button("Technical report (Markdown)", report, file_name=f"{stem}_report.md", mime="text/markdown")


def show_analysis(out: dict[str, Any], clip_path: str) -> None:
    results: list[Result] = out["results"]
    if out.get("message"):
        st.warning(out["message"])
        for result in results:
            for note in result.notes:
                st.caption(note)
        return
    if out.get("chart_png"):
        st.image(out["chart_png"], caption="Speed over the clip", use_container_width=True)
    show_ai_review(out)
    for result in results:
        st.subheader(result.module.replace("_", " ").capitalize())
        rows = [{"metric": LABELS.get(key, key), "value": format_value(value)} for key, value in result.metrics.items()]
        st.table(rows)
        for note in result.notes:
            st.caption(note)
        frames = [(label, out["key_frames"].get((result.module, label))) for label in result.key_frames]
        frames = [(label, img) for label, img in frames if img is not None]
        if frames:
            columns = st.columns(min(4, len(frames)))
            for column, (label, img) in zip(columns * 10, frames):
                column.image(img, caption=label, use_container_width=True)
        if result.details:
            with st.expander(f"Details by length ({len(result.details)} results)"):
                for detail in result.details:
                    st.markdown(f"**{detail.module.replace('_', ' ')}**")
                    rows = [{"metric": LABELS.get(k, k), "value": format_value(v)} for k, v in detail.metrics.items() if v is not None]
                    if rows:
                        st.table(rows)
                    for note in detail.notes:
                        if note.startswith(f"{detail.module.split('_')[0]}:"):
                            st.caption(note)
                    frames = [(label, out["key_frames"].get((detail.module, label))) for label in detail.key_frames]
                    frames = [(label, img) for label, img in frames if img is not None]
                    if frames:
                        columns = st.columns(min(4, len(frames)))
                        for column, (label, img) in zip(columns * 10, frames):
                            column.image(img, caption=label, use_container_width=True)
    st.subheader("Downloads")
    stem = os.path.splitext(st.session_state.get("upload_name", "clip"))[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.download_button("Report card (PNG)", out["card_png"], file_name=f"{stem}_report.png", mime="image/png")
    if out.get("pdf"):
        c2.download_button("Report (PDF)", out["pdf"], file_name=f"{stem}_report.pdf", mime="application/pdf")
    else:
        c2.caption("PDF needs matplotlib (pip install matplotlib)")
    c3.download_button("Numbers (JSON)", out["json"], file_name=f"{stem}_results.json", mime="application/json")
    if out.get("video"):
        with open(out["video"], "rb") as handle:
            data = handle.read()
        c4.download_button("Overlay video (MP4)", data, file_name=os.path.basename(out["video"]), mime="video/mp4")
        st.video(data)
    elif out.get("video_error"):
        c4.caption(f"Overlay video failed: {out['video_error']}")


def team_page(title: str = "SSAC SplitSecond") -> None:
    """The team page: Analyze one clip, Compare up to four, Saved results."""
    st.set_page_config(page_title=title, layout="wide")
    st.title(title)
    options = sidebar_options()
    tab_one, tab_compare, tab_saved = st.tabs(["Analyze", "Compare", "Saved results"])
    with tab_one:
        analyze_tab(options)
    with tab_compare:
        compare_tab(options)
    with tab_saved:
        saved_tab()


def analyze_tab(options: dict[str, Any]) -> None:
    phase_label = st.radio("What is in the video?", list(PHASES), horizontal=True)
    phase_key, expected_view = PHASES[phase_label]
    path = upload_clip()
    if path is None:
        st.info("Upload a clip to start.")
        return
    calibration_panel(path, options)
    if st.session_state.get("use_ai"):
        c1, c2 = st.columns([1, 2])
        st.session_state["swimmer"] = c1.text_input("Swimmer (for history)", value=st.session_state.get("swimmer", ""), key="swimmer_for_ai")
        st.session_state["ai_comments"] = c2.text_input("Coach or swimmer comments (optional)", value=st.session_state.get("ai_comments", ""))
    if st.button("Analyze", type="primary"):
        with st.spinner("Analyzing..."):
            st.session_state["results"] = run_analysis(path, phase_key, options, st.session_state.get("make_video", True),
                                                       expected_view)
            st.session_state["results_phase"] = phase_key
    if "results" in st.session_state:
        show_analysis(st.session_state["results"], path)
        save_panel(st.session_state["results"], st.session_state.get("results_phase", phase_key), options)


def save_panel(out: dict[str, Any], phase_key: str, options: dict[str, Any]) -> None:
    """Save the numbers on screen to the database."""
    from core.db import save_run

    results = out.get("results") or []
    if not results or out.get("message"):
        return
    st.subheader("Save to database")
    c1, c2 = st.columns([2, 1])
    swimmer = c1.text_input("Swimmer", value=st.session_state.get("swimmer", ""), key="swimmer_name")
    st.session_state["swimmer"] = swimmer
    if c2.button("Save these results"):
        run_id = save_run(results, clip_name=st.session_state.get("upload_name", "clip"), swimmer=swimmer,
                          phase=phase_key, course=str(options.get("course", "")))
        st.success(f"Saved as run {run_id}. See the Saved results tab.")


def compare_tab(options: dict[str, Any]) -> None:
    """Run the same phase on up to four clips and put the numbers side by side."""
    from core.db import save_run
    from core.render import compare_chart_png, speed_profile
    from core.lap import analyze_lap
    from modules import get_module

    st.caption("Same phase, up to four clips: different swimmers, or the same swimmer on different days.")
    phase_label = st.radio("What is in the videos?", list(PHASES), horizontal=True, key="compare_phase")
    phase_key, expected_view = PHASES[phase_label]
    uploads = st.file_uploader("Upload two to four clips", type=["mp4", "mov", "m4v", "avi"], accept_multiple_files=True,
                               key="compare_uploads")
    labels: list[str] = []
    if uploads:
        uploads = uploads[:4]
        cols = st.columns(len(uploads))
        for i, (col, up) in enumerate(zip(cols, uploads)):
            labels.append(col.text_input(f"Label for clip {i + 1}", value=os.path.splitext(up.name)[0], key=f"cmp_label_{i}"))
    if not uploads or len(uploads) < 2:
        st.info("Pick two to four clips.")
        return
    if st.button("Compare", type="primary"):
        runs = []
        bar = st.progress(0.0, text="Analyzing...")
        for i, up in enumerate(uploads):
            suffix = os.path.splitext(up.name)[1] or ".mp4"
            handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            handle.write(up.getbuffer())
            handle.close()
            opts = dict(options)
            if expected_view:
                opts["view"] = expected_view
            bar.progress(i / len(uploads), text=f"Analyzing {up.name}...")
            with open_clip(handle.name) as clip:
                results = analyze_lap(clip, opts) if phase_key == "lap" else [get_module(phase_key).run(clip, opts)]
                message = coverage_message(results, phase_key)
                profile = speed_profile(opts)
                align = None
                for r in results:
                    for key in ("entry", "breakout", "swim_start", "wall", "start"):
                        if key in r.key_frames and align is None:
                            align = clip.frame_to_time(int(r.key_frames[key]))
            runs.append({"label": labels[i], "clip": up.name, "results": results, "message": message, "profile": profile, "align": align})
        bar.progress(1.0, text="Done")
        st.session_state["compare"] = {"phase": phase_key, "runs": runs}
    if "compare" not in st.session_state:
        return
    comp = st.session_state["compare"]
    runs = comp["runs"]
    for run in runs:
        if run["message"]:
            st.warning(f"{run['label']}: {run['message']}")
    good = [r for r in runs if not r["message"]]
    if not good:
        return
    # one table per module: rows = metrics, columns = clips
    modules_seen: list[str] = []
    for run in good:
        for r in run["results"]:
            if r.module not in modules_seen:
                modules_seen.append(r.module)
    for module in modules_seen:
        st.subheader(module.replace("_", " ").capitalize())
        keys: list[str] = []
        per_clip: dict[str, dict[str, Any]] = {}
        for run in good:
            for r in run["results"]:
                if r.module == module:
                    per_clip[run["label"]] = r.metrics
                    for key in r.metrics:
                        if key not in keys:
                            keys.append(key)
        rows = []
        for key in keys:
            row = {"metric": LABELS.get(key, key)}
            for run in good:
                row[run["label"]] = format_value(per_clip.get(run["label"], {}).get(key))
            rows.append(row)
        st.table(rows)
    profiles = [(r["label"], r["profile"]) for r in good if r["profile"] is not None]
    if profiles:
        align_on = st.checkbox("Line the clips up on the first key moment (entry, breakout or wall)", value=True)
        chart = compare_chart_png(profiles, [r["align"] for r in good if r["profile"] is not None] if align_on else None)
        if chart:
            st.image(chart, caption="Speed, all clips", use_container_width=True)
    c1, c2 = st.columns([2, 1])
    swimmer = c1.text_input("Swimmer (leave blank if the clips are different swimmers)", key="compare_swimmer")
    if c2.button("Save all to database"):
        ids = [save_run(r["results"], clip_name=r["clip"], swimmer=swimmer or r["label"], phase=comp["phase"],
                        course=str(options.get("course", ""))) for r in good]
        st.success(f"Saved runs {', '.join(str(i) for i in ids)}.")


def saved_tab() -> None:
    from core.db import DB_PATH, delete_run, export_csv, load_metrics, load_runs

    runs = load_runs()
    st.caption(f"Database file: {DB_PATH}")
    if not runs:
        st.info("Nothing saved yet. Analyze a clip and press Save these results.")
        return
    st.table([{k: ("" if v is None else v) for k, v in r.items()} for r in runs])
    pick = st.selectbox("Show the numbers of run", [r["id"] for r in runs], format_func=lambda i: f"run {i}")
    rows = load_metrics(pick)
    st.table([{"module": r["module"], "metric": LABELS.get(r["metric"], r["metric"]), "value": format_value(r["value"])} for r in rows])
    c1, c2 = st.columns(2)
    c1.download_button("Download everything as CSV", export_csv(), file_name="splitsecond_results.csv", mime="text/csv")
    if c2.button(f"Delete run {pick}"):
        delete_run(pick)
        st.rerun()


def module_page(modules: list[AnalysisModule], title: str) -> None:
    """A complete page for one module: upload, run, show the numbers."""
    st.set_page_config(page_title=title, layout="wide")
    st.title(title)
    options = sidebar_options()
    path = upload_clip()
    if path is None:
        st.info("Upload a clip to start. To make a test clip run: python tools/make_sample_clip.py")
        return
    calibration_panel(path, options)
    if not st.button("Analyze", type="primary"):
        return
    results: list[Result] = []
    with st.spinner("Analyzing..."):
        with open_clip(path) as clip:
            st.caption(str(clip))
            for module in modules:
                result = module.run(clip, options)
                results.append(result)
                show_result(result, clip)
    show_advice(results)
    st.download_button("Download JSON", to_json(results, clip_path=path), file_name="results.json", mime="application/json")
