"""Small numeric helpers shared by the modules: smoothing, peaks, interpolation.

Only numpy is used so this runs anywhere the rest of the project runs.
"""

from __future__ import annotations

import numpy as np


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average. NaN values are ignored (treated as missing)."""
    values = np.asarray(values, dtype=float)
    window = max(1, int(window))
    if window == 1 or values.size == 0:
        return values.copy()
    if window % 2 == 0:
        window += 1
    if window > values.size:
        window = values.size if values.size % 2 == 1 else max(1, values.size - 1)
    ok = np.isfinite(values).astype(float)
    filled = np.where(np.isfinite(values), values, 0.0)
    kernel = np.ones(window)
    num = np.convolve(filled, kernel, mode="same")
    den = np.convolve(ok, kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den == 0] = np.nan
    return out


def median_filter(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    if window == 1 or values.size < window:
        return values.copy()
    half = window // 2
    padded = np.pad(values, half, mode="edge")
    out = np.empty_like(values)
    for i in range(values.size):
        chunk = padded[i : i + window]
        chunk = chunk[np.isfinite(chunk)]
        out[i] = np.median(chunk) if chunk.size else np.nan
    return out


def fill_gaps(values: np.ndarray) -> np.ndarray:
    """Linear interpolation over NaN gaps; edges are held at the nearest value."""
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(values)
    if ok.sum() == 0:
        return values.copy()
    if ok.all():
        return values.copy()
    idx = np.arange(values.size)
    return np.interp(idx, idx[ok], values[ok])


def fill_short_gaps(values: np.ndarray, max_gap: int) -> np.ndarray:
    """Like fill_gaps, but runs of NaN longer than max_gap samples stay NaN."""
    values = np.asarray(values, dtype=float)
    filled = fill_gaps(values)
    ok = np.isfinite(values)
    out = filled.copy()
    start = None
    for i in range(values.size + 1):
        missing = i < values.size and not ok[i]
        if missing and start is None:
            start = i
        elif not missing and start is not None:
            if i - start > max_gap:
                out[start:i] = np.nan
            start = None
    return out


def derivative(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Central-difference derivative of values with respect to times."""
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    if values.size < 2:
        return np.zeros_like(values)
    return np.gradient(values, times)


def find_peaks(values: np.ndarray, min_distance: int = 1, min_prominence: float = 0.0) -> list[int]:
    """Indexes of local maxima with a simple prominence test. No scipy needed."""
    values = np.asarray(values, dtype=float)
    n = values.size
    if n < 3:
        return []
    candidates = [i for i in range(1, n - 1) if values[i] > values[i - 1] and values[i] >= values[i + 1]]
    peaks: list[int] = []
    for i in candidates:
        if min_prominence > 0:
            left = values[max(0, i - min_distance * 2) : i]
            right = values[i + 1 : i + 1 + min_distance * 2]
            base = max(left.min() if left.size else values[i], right.min() if right.size else values[i])
            if values[i] - base < min_prominence:
                continue
        if peaks and i - peaks[-1] < min_distance:
            if values[i] > values[peaks[-1]]:
                peaks[-1] = i
            continue
        peaks.append(i)
    return peaks


def count_oscillations(values: np.ndarray, fps: float, min_hz: float = 0.8, max_hz: float = 4.0) -> tuple[int, list[int]]:
    """Count the cycles in a roughly periodic signal (kicks, strokes).

    The signal is detrended with a slow moving average, then peaks at least
    1/max_hz apart with a prominence above a fraction of the signal spread
    are counted. Returns (count, peak indexes).
    """
    values = np.asarray(values, dtype=float)
    values = fill_gaps(values)
    if values.size < 5 or not np.isfinite(values).any():
        return 0, []
    slow = moving_average(values, int(round(fps / min_hz)))
    fast = moving_average(values - slow, max(1, int(round(fps / (max_hz * 4)))))
    fast = np.nan_to_num(fast)
    spread = np.percentile(fast, 90) - np.percentile(fast, 10)
    if spread <= 1e-9:
        return 0, []
    peaks = find_peaks(fast, min_distance=max(1, int(round(fps / max_hz))), min_prominence=0.25 * spread)
    return len(peaks), peaks


def crossing_time(times: np.ndarray, values: np.ndarray, level: float, rising: bool = True) -> float | None:
    """First time the series crosses `level`, linearly interpolated. None if never."""
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    for i in range(1, values.size):
        a, b = values[i - 1], values[i]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if rising and a < level <= b or (not rising and a > level >= b):
            if b == a:
                return float(times[i])
            frac = (level - a) / (b - a)
            return float(times[i - 1] + frac * (times[i] - times[i - 1]))
    return None


def per_second(times: np.ndarray, values: np.ndarray) -> list[tuple[float, float]]:
    """Average of `values` in each whole second of `times`: [(second, mean), ...]."""
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    out: list[tuple[float, float]] = []
    if times.size == 0:
        return out
    for s in range(int(np.floor(times.min())), int(np.ceil(times.max())) + 1):
        pick = (times >= s) & (times < s + 1) & np.isfinite(values)
        if pick.any():
            out.append((float(s), float(values[pick].mean())))
    return out


def dominant_period(values: np.ndarray, fps: float, min_period_s: float, max_period_s: float) -> float | None:
    """Period (seconds) of the strongest rhythm in a signal, by autocorrelation. None if no rhythm."""
    values = np.asarray(values, dtype=float)
    values = fill_gaps(values)
    n = values.size
    if n < 8 or not np.isfinite(values).any():
        return None
    slow = moving_average(values, int(round(fps * max_period_s * 1.5)))
    x = np.nan_to_num(values - slow)
    x = x - x.mean()
    if np.allclose(x, 0):
        return None
    ac = np.correlate(x, x, mode="full")[n - 1 :]
    if ac[0] <= 0:
        return None
    ac = ac / ac[0]
    lo = max(1, int(round(min_period_s * fps)))
    hi = min(n - 2, int(round(max_period_s * fps)))
    if hi <= lo:
        return None
    peaks = [lag for lag in range(lo, hi + 1) if ac[lag] >= ac[lag - 1] and ac[lag] >= ac[lag + 1] and ac[lag] > 0.1]
    if not peaks:
        return None
    strongest = max(ac[lag] for lag in peaks)
    # the shortest lag that is still a strong rhythm is the stroke; longer
    # lags are usually two strokes (left arm, right arm)
    best = next(lag for lag in peaks if ac[lag] >= 0.4 * strongest)
    return best / fps


def count_rhythm(values: np.ndarray, fps: float, min_period_s: float, max_period_s: float) -> tuple[int, list[int], float | None]:
    """Count the cycles of a rhythmic signal: (count, peak indexes, period_s).

    The period is found first by autocorrelation, then peaks at least 0.7 of a
    period apart are counted. If no rhythm is found, falls back to plain
    peak counting inside the allowed range.
    """
    values = np.asarray(values, dtype=float)
    period = dominant_period(values, fps, min_period_s, max_period_s)
    if period is None:
        count, peaks = count_oscillations(values, fps, min_hz=1.0 / max_period_s, max_hz=1.0 / min_period_s)
        return count, peaks, None
    filled = fill_gaps(values)
    slow = moving_average(filled, int(round(fps * period * 2)))
    fast = np.nan_to_num(moving_average(filled - slow, max(1, int(round(fps * period / 6)))))
    spread = np.percentile(fast, 90) - np.percentile(fast, 10)
    peaks = find_peaks(fast, min_distance=max(1, int(round(0.7 * period * fps))), min_prominence=0.1 * spread)
    # with a clear rhythm the count is the window length over the period;
    # the peaks are kept for key frames and are usually within one of it
    by_period = int(round(values.size / fps / period))
    count = max(len(peaks), by_period)
    return count, peaks, period
