"""Duration-aware alignment data model and decision logic.

This module is the core of the ``foreign_whispers`` library.  It answers the
central question of the dubbing pipeline: *how do we fit a target-language
translation into the same time window as the original source-language speech?*

The module provides:

- ``SegmentMetrics`` — measures the timing mismatch for each segment.
- ``decide_action`` — per-segment policy that chooses accept / stretch / shift / retry / fail.
- ``global_align`` — greedy left-to-right pass that schedules all segments
  on a shared timeline, tracking cumulative drift from gap shifts.

No external dependencies — stdlib only.
"""
import dataclasses
import re
import unicodedata
from enum import Enum


def _count_syllables(text: str) -> int:
    """Count syllables in target-language text via vowel-cluster counting.

    Designed for Romance languages (Spanish, French, Italian, Portuguese).
    Strips accents then counts contiguous vowel runs. Each run = one syllable.
    Returns at least 1 for any non-empty text so the rate never divides by zero.
    """
    # Normalise: decompose accented chars, keep only ASCII letters + spaces
    nfkd = unicodedata.normalize("NFKD", text.lower())
    ascii_text = "".join(c for c in nfkd if not unicodedata.combining(c))
    clusters = re.findall(r"[aeiou]+", ascii_text)
    return max(1, len(clusters))


_SYLLABLE_RATE = 4.5  # syllables per second for Romance languages

# Linear regression fitted on real Chatterbox multilingual TTS output:
#   duration_s = 0.1735 * syllables + 0.8301
# The intercept captures fixed per-segment overhead (model onset, leading
# silence, breath); the slope corresponds to ~5.76 syl/s during sustained
# speech — faster than the raw 4.5 rate because the intercept absorbs the
# startup cost that the old purely-rate-based model bundled in.
_SYL_SLOPE = 0.1735
_SYL_INTERCEPT = 0.8301


def _estimate_duration(text: str) -> float:
    """Estimate TTS duration in seconds.

    Uses a linear model fitted on real Chatterbox output, plus a small bonus
    for sentence-final punctuation which produces audible trailing silence.
    """
    if not text or not text.strip():
        return 0.0
    syllables = _count_syllables(text)
    base = _SYL_SLOPE * syllables + _SYL_INTERCEPT
    # Sentence-final punctuation adds ~0.15s of trailing silence each.
    pause_bonus = 0.15 * sum(text.count(p) for p in ".!?")
    return base + pause_bonus


@dataclasses.dataclass
class SegmentMetrics:
    """Timing measurements for one source/target transcript segment pair.

    For each segment we know the original source-language duration (from Whisper
    timestamps) and the translated target-language text.  The question is:
    *will the target-language TTS audio fit inside the source time window?*

    We estimate the TTS duration using a syllable-rate heuristic
    (~4.5 syllables/second for Romance languages) and derive three key numbers:

    Attributes:
        index: Zero-based segment position in the transcript.
        source_start: Source-language segment start time (seconds).
        source_end: Source-language segment end time (seconds).
        source_duration_s: ``source_end - source_start``.
        source_text: Original source-language text.
        translated_text: Target-language translation.
        src_char_count: Character count of the source text.
        tgt_char_count: Character count of the target text.
        predicted_tts_s: Estimated TTS duration (syllables / 4.5).
        predicted_stretch: Ratio ``predicted_tts_s / source_duration_s``.
            A value of 1.3 means the target-language audio is predicted to be
            30% longer than the available window.
        overflow_s: How many seconds the target-language audio exceeds the
            window (zero when it fits).
    """
    index:             int
    source_start:      float
    source_end:        float
    source_duration_s: float
    source_text:       str
    translated_text:   str
    src_char_count:    int
    tgt_char_count:    int
    predicted_tts_s:   float = dataclasses.field(init=False)
    predicted_stretch: float = dataclasses.field(init=False)
    overflow_s:        float = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.predicted_tts_s = _estimate_duration(self.translated_text)
        self.predicted_stretch = (
            self.predicted_tts_s / self.source_duration_s
            if self.source_duration_s > 0 else 1.0
        )
        self.overflow_s = max(0.0, self.predicted_tts_s - self.source_duration_s)


class AlignAction(str, Enum):
    """Decision outcomes for the per-segment alignment policy.

    Each segment gets exactly one action based on its ``predicted_stretch``:

    - ``ACCEPT`` — fits within 10% of the original duration, no change needed.
    - ``MILD_STRETCH`` — 10–40% over; apply pyrubberband time-stretch.
    - ``GAP_SHIFT`` — 40–80% over but adjacent silence can absorb the overflow.
    - ``REQUEST_SHORTER`` — 80–150% over; needs a shorter translation (P8).
    - ``FAIL`` — >150% over; no fix available, log and fall back to silence.
    """
    ACCEPT          = "accept"
    MILD_STRETCH    = "mild_stretch"
    GAP_SHIFT       = "gap_shift"
    REQUEST_SHORTER = "request_shorter"
    FAIL            = "fail"


@dataclasses.dataclass
class AlignedSegment:
    """A segment with its scheduled position on the global timeline.

    Produced by ``global_align``.  The ``scheduled_start`` and
    ``scheduled_end`` incorporate cumulative drift from earlier gap shifts,
    so they may differ from the original Whisper timestamps.

    Attributes:
        index: Segment position (matches ``SegmentMetrics.index``).
        original_start: Whisper start time (seconds).
        original_end: Whisper end time (seconds).
        scheduled_start: Start time after global alignment (seconds).
        scheduled_end: End time after global alignment (seconds).
        text: Target-language translated text for this segment.
        action: The ``AlignAction`` chosen by ``decide_action``.
        gap_shift_s: Seconds borrowed from adjacent silence (0.0 if none).
        stretch_factor: Speed factor for pyrubberband (1.0 = no stretch).
    """
    index:           int
    original_start:  float
    original_end:    float
    scheduled_start: float
    scheduled_end:   float
    text:            str
    action:          AlignAction
    gap_shift_s:     float = 0.0
    stretch_factor:  float = 1.0


def decide_action(m: SegmentMetrics, available_gap_s: float = 0.0) -> AlignAction:
    """Choose the alignment action for a single segment.

    Maps the predicted stretch factor to one of five actions using fixed
    thresholds.  ``GAP_SHIFT`` additionally requires that enough silence
    follows the segment to absorb the overflow.

    Thresholds::

        predicted_stretch   Action            Condition
        ─────────────────   ────────────────  ─────────────────────────
        <= 1.1              ACCEPT            fits naturally
        1.1 – 1.4          MILD_STRETCH      pyrubberband safe range
        1.4 – 1.8          GAP_SHIFT         only if gap >= overflow
        1.8 – 2.5          REQUEST_SHORTER   needs shorter translation
        > 2.5              FAIL              unfixable

    Args:
        m: Timing metrics for one segment.
        available_gap_s: Silence duration (seconds) after this segment,
            from VAD.  Defaults to 0.0 (no gap available).

    Returns:
        The ``AlignAction`` to apply.
    """
    sf = m.predicted_stretch
    if sf <= 1.1:
        return AlignAction.ACCEPT
    if sf <= 1.4:
        return AlignAction.MILD_STRETCH
    if sf <= 1.8 and available_gap_s >= m.overflow_s:
        return AlignAction.GAP_SHIFT
    if sf <= 2.5:
        return AlignAction.REQUEST_SHORTER
    return AlignAction.FAIL


def compute_segment_metrics(
    en_transcript: dict,
    es_transcript: dict,
) -> list[SegmentMetrics]:
    """Pair source and target segments and compute per-segment timing metrics.

    Zips the ``"segments"`` lists from both transcripts positionally
    (segment 0 ↔ segment 0, etc.) and builds a ``SegmentMetrics`` for each
    pair.  The source segment provides the time window; the target segment
    provides the text whose TTS duration we need to predict.

    Args:
        en_transcript: Source-language Whisper output dict with
            ``{"segments": [{"start", "end", "text"}, ...]}``.
        es_transcript: Target-language translation dict with the same structure.

    Returns:
        List of ``SegmentMetrics``, one per paired segment.  If the transcripts
        have different lengths, the shorter one determines the output length.
    """
    metrics = []
    for i, (en_seg, es_seg) in enumerate(
        zip(en_transcript.get("segments", []), es_transcript.get("segments", []))
    ):
        src_text = en_seg["text"].strip()
        tgt_text = es_seg["text"].strip()
        metrics.append(SegmentMetrics(
            index             = i,
            source_start      = en_seg["start"],
            source_end        = en_seg["end"],
            source_duration_s = en_seg["end"] - en_seg["start"],
            source_text       = src_text,
            translated_text   = tgt_text,
            src_char_count    = len(src_text),
            tgt_char_count    = len(tgt_text),
        ))
    return metrics


def global_align(
    metrics:         list[SegmentMetrics],
    silence_regions: list[dict],
    max_stretch:     float = 1.4,
) -> list[AlignedSegment]:
    """Greedy left-to-right global alignment of dubbed segments.

    Segments are timed independently by ``decide_action`` (P7), but they are
    sequential — if segment 5 borrows 0.3s from a silence gap, every segment
    after it shifts by 0.3s.  This function tracks that cumulative drift.

    Algorithm (single pass, O(n)):

    1. For each segment, call ``decide_action(m, available_gap_s)`` where
       *available_gap_s* comes from VAD silence regions after this segment.
    2. Based on the action:

       - ``GAP_SHIFT`` — the segment expands into the silence after it
         (``gap_shift = overflow_s``).
       - ``MILD_STRETCH`` — time-stretch capped at *max_stretch* (default 1.4x).
       - ``ACCEPT``, ``REQUEST_SHORTER``, ``FAIL`` — no modification.

    3. Schedule the segment with cumulative drift applied::

           scheduled_start = original_start + cumulative_drift
           scheduled_end   = scheduled_start + original_duration + gap_shift

    4. Every ``gap_shift`` adds to *cumulative_drift*, pushing all subsequent
       segments forward.

    Limitations:

    - **Greedy** — never looks ahead.  If segment 10 has a huge overflow and
      segment 9 has a large silence gap, it will not save that gap for
      segment 10.
    - **No backtracking** — once a decision is made, it is final.
    - A dynamic-programming or constraint-solver approach would produce
      better schedules, but this is the baseline to start from.

    Args:
        metrics: Per-segment timing metrics from ``compute_segment_metrics``.
        silence_regions: VAD output — list of ``{"start_s", "end_s", "label"}``
            dicts.  Pass ``[]`` if VAD is unavailable (gap_shift disabled).
        max_stretch: Upper bound for ``MILD_STRETCH`` speed factor.

    Returns:
        One ``AlignedSegment`` per input metric, in order.
    """
    def _silence_after(end_s: float) -> float:
        for r in silence_regions:
            if r.get("label") == "silence" and r["start_s"] >= end_s - 0.1:
                return r["end_s"] - r["start_s"]
        return 0.0

    aligned, cumulative_drift = [], 0.0

    for m in metrics:
        action    = decide_action(m, available_gap_s=_silence_after(m.source_end))
        gap_shift = 0.0
        stretch   = 1.0

        if action == AlignAction.GAP_SHIFT:
            gap_shift = m.overflow_s
        elif action == AlignAction.MILD_STRETCH:
            stretch = min(m.predicted_stretch, max_stretch)
        # ACCEPT, REQUEST_SHORTER, FAIL → stretch stays at 1.0

        sched_start = m.source_start + cumulative_drift
        sched_end   = sched_start + m.source_duration_s + gap_shift

        aligned.append(AlignedSegment(
            index           = m.index,
            original_start  = m.source_start,
            original_end    = m.source_end,
            scheduled_start = sched_start,
            scheduled_end   = sched_end,
            text            = m.translated_text,
            action          = action,
            gap_shift_s     = gap_shift,
            stretch_factor  = stretch,
        ))

        cumulative_drift += gap_shift

    return aligned


def global_align_dp(
    metrics:         list[SegmentMetrics],
    silence_regions: list[dict],
    max_stretch:     float = 1.4,
    drift_weight:    float = 0.5,
    stretch_weight:  float = 1.0,
    retry_weight:    float = 1.0,
    fail_weight:     float = 5.0,
    drift_bucket_s:  float = 0.1,
    max_drift_s:     float = 10.0,
) -> list[AlignedSegment]:
    """DP-based global alignment that beats the greedy scheduler.

    Greedy ``global_align`` makes per-segment decisions with fixed thresholds.
    This routine searches the space of (action, drift) sequences using
    Bellman dynamic programming on a discretised cumulative-drift state, and
    picks the schedule that minimises a weighted cost::

        cost = stretch_weight * Σ(stretch_factor − 1)²
             + drift_weight   * total_cumulative_drift
             + retry_weight   * n_request_shorter
             + fail_weight    * n_fail

    For each segment the optimiser chooses among: ``ACCEPT`` (no overflow),
    ``MILD_STRETCH`` (capped at *max_stretch*), ``GAP_SHIFT`` (when the
    silence after the segment can absorb the overflow), ``REQUEST_SHORTER``,
    and ``FAIL``. State space is O(n · D) where D is the number of drift
    buckets — fast enough for typical 100-300 segment clips.

    Args:
        metrics: Per-segment timing metrics from ``compute_segment_metrics``.
        silence_regions: VAD silence list, same format as ``global_align``.
        max_stretch: Upper bound for ``MILD_STRETCH`` speed factor.
        drift_weight: Cost per second of cumulative drift.
        stretch_weight: Cost coefficient on the squared stretch term.
        retry_weight: Cost per ``REQUEST_SHORTER`` action.
        fail_weight: Cost per ``FAIL`` action.
        drift_bucket_s: Discretisation step for the drift state (seconds).
        max_drift_s: Hard cap on cumulative drift the DP will explore.

    Returns:
        One ``AlignedSegment`` per metric, in order — same shape as
        ``global_align`` so it can be used as a drop-in replacement.
    """
    n = len(metrics)
    if n == 0:
        return []

    def _silence_after(end_s: float) -> float:
        for r in silence_regions:
            if r.get("label") == "silence" and r["start_s"] >= end_s - 0.1:
                return r["end_s"] - r["start_s"]
        return 0.0

    n_buckets = int(max_drift_s / drift_bucket_s) + 1
    INF = float("inf")

    # dp[i][b] = minimum cumulative cost to schedule segments 0..i-1 with
    # cumulative drift == b * drift_bucket_s. parent stores the back-pointer
    # so we can reconstruct the action sequence.
    dp = [[INF] * n_buckets for _ in range(n + 1)]
    parent: list[list[tuple | None]] = [[None] * n_buckets for _ in range(n + 1)]
    dp[0][0] = 0.0

    for i, m in enumerate(metrics):
        gap = _silence_after(m.source_end)
        # Per-segment costs that don't depend on incoming drift.
        # ACCEPT — only if the segment already fits comfortably.
        accept_ok = m.predicted_stretch <= 1.1
        # MILD_STRETCH — only valid when stretching at the cap actually fits
        # the segment (otherwise it's just clipped audio).
        stretch_ok = 1.0 < m.predicted_stretch <= max_stretch
        stretch_factor = m.predicted_stretch if stretch_ok else 1.0
        stretch_cost = stretch_weight * (stretch_factor - 1.0) ** 2
        # GAP_SHIFT — needs the silence to fully absorb the overflow.
        gap_shift_ok = m.overflow_s > 0 and gap >= m.overflow_s
        gap_shift_cost = drift_weight * m.overflow_s
        # REQUEST_SHORTER — only sensible if stretch is between 1.4 and 2.5.
        retry_ok = 1.4 < m.predicted_stretch <= 2.5
        # FAIL — last resort for anything beyond 2.5×.
        fail_needed = m.predicted_stretch > 2.5

        for b in range(n_buckets):
            if dp[i][b] == INF:
                continue
            base = dp[i][b]

            # Try each action and relax the next state.
            def relax(new_b: int, added_cost: float, action: AlignAction,
                      gap_shift: float, stretch: float) -> None:
                if new_b >= n_buckets:
                    return
                new_cost = base + added_cost
                if new_cost < dp[i + 1][new_b]:
                    dp[i + 1][new_b] = new_cost
                    parent[i + 1][new_b] = (b, action, gap_shift, stretch)

            if accept_ok:
                relax(b, 0.0, AlignAction.ACCEPT, 0.0, 1.0)

            if stretch_ok:
                relax(b, stretch_cost, AlignAction.MILD_STRETCH, 0.0, stretch_factor)

            if gap_shift_ok:
                added_buckets = max(1, int(round(m.overflow_s / drift_bucket_s)))
                relax(b + added_buckets, gap_shift_cost,
                      AlignAction.GAP_SHIFT, m.overflow_s, 1.0)

            if retry_ok:
                relax(b, retry_weight, AlignAction.REQUEST_SHORTER, 0.0, 1.0)

            if fail_needed:
                relax(b, fail_weight, AlignAction.FAIL, 0.0, 1.0)

    # Pick the best end-state across all drift buckets.
    best_b = min(range(n_buckets), key=lambda b: dp[n][b])
    if dp[n][best_b] == INF:
        # No feasible schedule under the constraints — fall back to greedy.
        return global_align(metrics, silence_regions, max_stretch)

    # Reconstruct the action sequence by walking the parent pointers backwards.
    chosen: list[tuple[AlignAction, float, float]] = [None] * n  # type: ignore
    b = best_b
    for i in range(n, 0, -1):
        prev = parent[i][b]
        if prev is None:
            return global_align(metrics, silence_regions, max_stretch)
        prev_b, action, gap_shift, stretch = prev
        chosen[i - 1] = (action, gap_shift, stretch)
        b = prev_b

    aligned: list[AlignedSegment] = []
    cumulative_drift = 0.0
    for m, (action, gap_shift, stretch) in zip(metrics, chosen):
        sched_start = m.source_start + cumulative_drift
        sched_end = sched_start + m.source_duration_s + gap_shift
        aligned.append(AlignedSegment(
            index           = m.index,
            original_start  = m.source_start,
            original_end    = m.source_end,
            scheduled_start = sched_start,
            scheduled_end   = sched_end,
            text            = m.translated_text,
            action          = action,
            gap_shift_s     = gap_shift,
            stretch_factor  = stretch,
        ))
        cumulative_drift += gap_shift

    return aligned
