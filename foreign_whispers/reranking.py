"""Deterministic failure analysis and translation re-ranking stubs.

The failure analysis function uses simple threshold rules derived from
SegmentMetrics.  The translation re-ranking function is a **student assignment**
— see the docstring for inputs, outputs, and implementation guidance.
"""

import dataclasses
import logging
import re

logger = logging.getLogger(__name__)


# Spanish phrase contractions: longer form → shorter equivalent.
# Each entry preserves meaning while shaving characters.
_CONTRACTIONS: dict[str, str] = {
    "en este momento": "ahora",
    "en estos momentos": "ahora",
    "en la actualidad": "hoy",
    "hoy en día": "hoy",
    "a pesar de que": "aunque",
    "a pesar de eso": "aún así",
    "con el fin de": "para",
    "con el objetivo de": "para",
    "con el propósito de": "para",
    "con la intención de": "para",
    "debido a que": "porque",
    "puesto que": "porque",
    "ya que": "porque",
    "dado que": "porque",
    "por consiguiente": "así",
    "por lo tanto": "así",
    "es decir": "o sea",
    "sin embargo": "pero",
    "no obstante": "pero",
    "a fin de cuentas": "al final",
    "en el caso de que": "si",
    "en caso de que": "si",
    "siempre y cuando": "si",
    "más o menos": "casi",
    "una gran cantidad de": "muchos",
    "un gran número de": "muchos",
    "la mayor parte de": "la mayoría de",
    "todos y cada uno": "todos",
    "tener en cuenta": "considerar",
    "tomar en consideración": "considerar",
    "llevar a cabo": "hacer",
    "darse cuenta de": "ver",
    "hacer referencia a": "referirse a",
    "tener la posibilidad de": "poder",
    "tener la capacidad de": "poder",
    "estar en condiciones de": "poder",
    "está claro que": "claramente",
    "es evidente que": "claramente",
    "lo que sucede es que": "es que",
    "lo que pasa es que": "es que",
    "el día de hoy": "hoy",
    "de manera que": "así",
    "de tal manera que": "así",
    "de forma que": "así",
}

# Filler / hedging words that can be dropped without losing meaning.
_FILLERS: list[str] = [
    "básicamente",
    "esencialmente",
    "literalmente",
    "obviamente",
    "evidentemente",
    "francamente",
    "honestamente",
    "claramente",
    "ciertamente",
    "definitivamente",
    "en realidad",
    "de hecho",
    "por cierto",
    "a decir verdad",
    "como sea",
    "pues bien",
    "bueno",
    "este",
    "o sea",
]


def _apply_contractions(text: str) -> tuple[str, int]:
    """Replace every long phrase with its shorter equivalent. Case-insensitive
    match, but the replacement is lowercase. Returns (new_text, n_replacements)."""
    n = 0
    out = text
    for long_form, short_form in _CONTRACTIONS.items():
        pattern = re.compile(re.escape(long_form), re.IGNORECASE)
        new_out, count = pattern.subn(short_form, out)
        if count:
            out = new_out
            n += count
    return out, n


def _strip_fillers(text: str) -> tuple[str, int]:
    """Remove filler/hedge words. Returns (new_text, n_removed)."""
    n = 0
    out = text
    for filler in _FILLERS:
        # Match the filler as a standalone word with optional surrounding
        # commas/whitespace, so "básicamente," and " obviamente " both go.
        pattern = re.compile(
            r"(?:^|\s|,)\s*" + re.escape(filler) + r"\s*(?:,|\s|$)",
            re.IGNORECASE,
        )
        new_out, count = pattern.subn(" ", out)
        if count:
            out = new_out
            n += count
    # Collapse multi-spaces and tidy punctuation.
    out = re.sub(r"\s+", " ", out).strip()
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    return out, n


@dataclasses.dataclass
class TranslationCandidate:
    """A candidate translation that fits a duration budget.

    Attributes:
        text: The translated text.
        char_count: Number of characters in *text*.
        brevity_rationale: Short explanation of what was shortened.
    """
    text: str
    char_count: int
    brevity_rationale: str = ""


@dataclasses.dataclass
class FailureAnalysis:
    """Diagnostic summary of the dominant failure mode in a clip.

    Attributes:
        failure_category: One of "duration_overflow", "cumulative_drift",
            "stretch_quality", or "ok".
        likely_root_cause: One-sentence description.
        suggested_change: Most impactful next action.
    """
    failure_category: str
    likely_root_cause: str
    suggested_change: str


def analyze_failures(report: dict) -> FailureAnalysis:
    """Classify the dominant failure mode from a clip evaluation report.

    Pure heuristic — no LLM needed.  The thresholds below match the policy
    bands defined in ``alignment.decide_action``.

    Args:
        report: Dict returned by ``clip_evaluation_report()``.  Expected keys:
            ``mean_abs_duration_error_s``, ``pct_severe_stretch``,
            ``total_cumulative_drift_s``, ``n_translation_retries``.

    Returns:
        A ``FailureAnalysis`` dataclass.
    """
    mean_err = report.get("mean_abs_duration_error_s", 0.0)
    pct_severe = report.get("pct_severe_stretch", 0.0)
    drift = abs(report.get("total_cumulative_drift_s", 0.0))
    retries = report.get("n_translation_retries", 0)

    if pct_severe > 20:
        return FailureAnalysis(
            failure_category="duration_overflow",
            likely_root_cause=(
                f"{pct_severe:.0f}% of segments exceed the 1.4x stretch threshold — "
                "translated text is consistently too long for the available time window."
            ),
            suggested_change="Implement duration-aware translation re-ranking (P8).",
        )

    if drift > 3.0:
        return FailureAnalysis(
            failure_category="cumulative_drift",
            likely_root_cause=(
                f"Total drift is {drift:.1f}s — small per-segment overflows "
                "accumulate because gaps between segments are not being reclaimed."
            ),
            suggested_change="Enable gap_shift in the global alignment optimizer (P9).",
        )

    if mean_err > 0.8:
        return FailureAnalysis(
            failure_category="stretch_quality",
            likely_root_cause=(
                f"Mean duration error is {mean_err:.2f}s — segments fit within "
                "stretch limits but the stretch distorts audio quality."
            ),
            suggested_change="Lower the mild_stretch ceiling or shorten translations.",
        )

    return FailureAnalysis(
        failure_category="ok",
        likely_root_cause="No dominant failure mode detected.",
        suggested_change="Review individual outlier segments if any remain.",
    )


def get_shorter_translations(
    source_text: str,
    baseline_es: str,
    target_duration_s: float,
    context_prev: str = "",
    context_next: str = "",
) -> list[TranslationCandidate]:
    """Return shorter translation candidates that fit *target_duration_s*.

    .. admonition:: Student Assignment — Duration-Aware Translation Re-ranking

       This function is intentionally a **stub that returns an empty list**.
       Your task is to implement a strategy that produces shorter
       target-language translations when the baseline translation is too long
       for the time budget.

       **Inputs**

       ============== ======== ==================================================
       Parameter      Type     Description
       ============== ======== ==================================================
       source_text    str      Original source-language segment text
       baseline_es    str      Baseline target-language translation (from argostranslate)
       target_duration_s float Time budget in seconds for this segment
       context_prev   str      Text of the preceding segment (for coherence)
       context_next   str      Text of the following segment (for coherence)
       ============== ======== ==================================================

       **Outputs**

       A list of ``TranslationCandidate`` objects, sorted shortest first.
       Each candidate has:

       - ``text``: the shortened target-language translation
       - ``char_count``: ``len(text)``
       - ``brevity_rationale``: short note on what was changed

       **Duration heuristic**: target-language TTS produces ~15 characters/second
       (or ~4.5 syllables/second for Romance languages).  So a 3-second budget
       ≈ 45 characters.

       **Approaches to consider** (pick one or combine):

       1. **Rule-based shortening** — strip filler words, use shorter synonyms
          from a lookup table, contract common phrases
          (e.g. "en este momento" → "ahora").
       2. **Multiple translation backends** — call argostranslate with
          paraphrased input, or use a second translation model, then pick
          the shortest output that preserves meaning.
       3. **LLM re-ranking** — use an LLM (e.g. via an API) to generate
          condensed alternatives.  This was the previous approach but adds
          latency, cost, and a runtime dependency.
       4. **Hybrid** — rule-based first, fall back to LLM only for segments
          that still exceed the budget.

       **Evaluation criteria**: the caller selects the candidate whose
       ``len(text) / 15.0`` is closest to ``target_duration_s``.

    Returns:
        Empty list (stub).  Implement to return ``TranslationCandidate`` items.
    """
    # Character budget: ~15 chars/sec for Spanish TTS.
    char_budget = max(1, int(target_duration_s * 15))

    candidates: dict[str, str] = {}  # text → rationale, dedup by text

    # Strategy 1: apply phrase contractions only.
    contracted, n_contractions = _apply_contractions(baseline_es)
    if n_contractions and contracted != baseline_es:
        candidates[contracted] = f"contracted {n_contractions} phrase(s)"

    # Strategy 2: strip filler/hedge words only.
    stripped, n_fillers = _strip_fillers(baseline_es)
    if n_fillers and stripped != baseline_es:
        candidates[stripped] = f"removed {n_fillers} filler(s)"

    # Strategy 3: both — contractions then filler removal.
    if n_contractions or n_fillers:
        combined, _ = _strip_fillers(contracted)
        if combined != baseline_es and combined not in candidates:
            candidates[combined] = (
                f"contracted {n_contractions} phrase(s), removed {n_fillers} filler(s)"
            )

    # Keep only candidates that are strictly shorter than the baseline AND
    # within the character budget. (Even if a candidate exceeds the budget,
    # it's still useful as a partial improvement — keep it but mark over-budget
    # ones lower priority by sorting purely on length.)
    out = [
        TranslationCandidate(
            text=text,
            char_count=len(text),
            brevity_rationale=rationale,
        )
        for text, rationale in candidates.items()
        if len(text) < len(baseline_es)
    ]

    # Shortest first.
    out.sort(key=lambda c: c.char_count)

    logger.info(
        "get_shorter_translations: budget=%.1fs (~%d chars), baseline=%d chars, "
        "produced %d candidate(s).",
        target_duration_s,
        char_budget,
        len(baseline_es),
        len(out),
    )
    return out
