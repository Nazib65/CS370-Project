"""Clip-level alignment quality metrics.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M8-align).
Imports from foreign_whispers.alignment — no other dependencies.
"""
import math
import statistics as _stats

from foreign_whispers.alignment import (
    AlignAction,
    AlignedSegment,
    SegmentMetrics,
    decide_action,
)


def _clamp01(x: float) -> float:
    """Clamp a float into the [0, 1] interval."""
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def clip_evaluation_report(
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
) -> dict:
    """Return a summary dict of alignment quality metrics for one clip.

    Keys:
        mean_abs_duration_error_s: Mean |predicted_tts_s - source_duration_s| per segment.
        pct_severe_stretch: % of aligned segments with stretch_factor > 1.4.
        n_gap_shifts: Number of segments resolved via gap-shift.
        n_translation_retries: Number of segments that required re-ranking.
        total_cumulative_drift_s: End-to-end drift introduced by gap-shifts.
    """
    if not metrics:
        return {
            "mean_abs_duration_error_s": 0.0,
            "pct_severe_stretch":        0.0,
            "n_gap_shifts":              0,
            "n_translation_retries":     0,
            "total_cumulative_drift_s":  0.0,
        }

    errors    = [abs(m.predicted_tts_s - m.source_duration_s) for m in metrics]
    n_severe  = sum(1 for a in aligned if a.stretch_factor > 1.4)
    n_shifted = sum(1 for a in aligned if a.action == AlignAction.GAP_SHIFT)
    n_retry   = sum(1 for m in metrics if decide_action(m) == AlignAction.REQUEST_SHORTER)
    drift     = (
        aligned[-1].scheduled_end - aligned[-1].original_end
        if aligned else 0.0
    )

    return {
        "mean_abs_duration_error_s": round(_stats.mean(errors), 3),
        "pct_severe_stretch":        round(100 * n_severe / max(len(metrics), 1), 1),
        "n_gap_shifts":              n_shifted,
        "n_translation_retries":     n_retry,
        "total_cumulative_drift_s":  round(drift, 3),
    }


def dubbing_scorecard(
    metrics: list[SegmentMetrics],
    aligned_segments: list[AlignedSegment],
    align_report: dict | None = None,
) -> dict:
    """Multi-dimensional dubbing quality scorecard.

    Returns five scores in ``[0, 1]`` (higher = better) plus an aggregate
    ``overall`` score (geometric mean — penalises any single weak dimension).

    Dimensions (computable from timing data alone, no audio needed):

    - ``timing_accuracy``  — how close predicted TTS durations match the source
      time windows. Derived from ``mean_abs_duration_error_s``.
    - ``stretch_quality``  — proportion of segments that don't need a severe
      time-stretch (>1.4×). Severe stretches sound unnatural.
    - ``drift_control``    — total cumulative drift relative to clip length.
      Heavy drift means the dub gets progressively out of sync with the video.
    - ``coverage``         — fraction of segments that pass the per-segment
      policy (not ``REQUEST_SHORTER`` or ``FAIL``). High coverage = the
      translator's output mostly fits the timing budget.
    - ``naturalness``      — inverse of the variance of per-segment speaking
      rate (chars/second). Low variance = consistent pace.

    Two further dimensions are placeholders that require the actual TTS audio
    (not provided to this function): ``intelligibility`` would do an STT
    round-trip and compute WER vs the translation, and ``semantic_fidelity``
    would compare embedding similarity between the source and a back-
    translation. Both are returned as ``None`` so callers can plug them in
    later without breaking the schema.
    """
    if not metrics:
        return {
            "timing_accuracy":   1.0,
            "stretch_quality":   1.0,
            "drift_control":     1.0,
            "coverage":          1.0,
            "naturalness":       1.0,
            "intelligibility":   None,
            "semantic_fidelity": None,
            "overall":           1.0,
            "n_segments":        0,
        }

    report = align_report if align_report is not None else clip_evaluation_report(metrics, aligned_segments)
    n = len(metrics)

    # ── 1. Timing accuracy: error of 0s → 1.0, error of 2s → ~0.33.
    err = report.get("mean_abs_duration_error_s", 0.0)
    timing_accuracy = 1.0 / (1.0 + err)

    # ── 2. Stretch quality: 1 − (fraction of severe stretches).
    pct_severe = report.get("pct_severe_stretch", 0.0)
    stretch_quality = _clamp01(1.0 - pct_severe / 100.0)

    # ── 3. Drift control: drift normalised by total clip duration.
    drift = abs(report.get("total_cumulative_drift_s", 0.0))
    clip_duration = max(
        sum(m.source_duration_s for m in metrics),
        1.0,  # avoid div-by-zero on degenerate inputs
    )
    drift_control = _clamp01(1.0 - drift / clip_duration)

    # ── 4. Coverage: fraction of segments that fit policy without retry/fail.
    bad_actions = {AlignAction.REQUEST_SHORTER, AlignAction.FAIL}
    n_ok = sum(1 for m in metrics if decide_action(m) not in bad_actions)
    coverage = n_ok / n

    # ── 5. Naturalness: low variance of speaking rate is good.
    # Speaking rate = chars / predicted_tts_s. Skip segments with no speech.
    rates = [
        m.tgt_char_count / m.predicted_tts_s
        for m in metrics
        if m.predicted_tts_s > 0.05 and m.tgt_char_count > 0
    ]
    if len(rates) >= 2:
        mean_rate = _stats.mean(rates)
        # Coefficient of variation is the natural normalisation here.
        cv = _stats.pstdev(rates) / mean_rate if mean_rate > 0 else 1.0
        # CV of 0 → naturalness 1.0; CV of 0.5 → ~0.67; CV of 1.0 → 0.5.
        naturalness = 1.0 / (1.0 + cv)
    else:
        naturalness = 1.0

    dims = [timing_accuracy, stretch_quality, drift_control, coverage, naturalness]
    # Geometric mean — any single weak dimension drags the overall score down.
    log_sum = sum(math.log(max(d, 1e-6)) for d in dims)
    overall = math.exp(log_sum / len(dims))

    return {
        "timing_accuracy":   round(timing_accuracy, 3),
        "stretch_quality":   round(stretch_quality, 3),
        "drift_control":     round(drift_control, 3),
        "coverage":          round(coverage, 3),
        "naturalness":       round(naturalness, 3),
        "intelligibility":   None,   # requires TTS audio + STT round-trip
        "semantic_fidelity": None,   # requires embedding model
        "overall":           round(overall, 3),
        "n_segments":        n,
    }
