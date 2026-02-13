"""
Response quality evaluation using heuristic scoring.

Evaluates LLM responses on four axes -- relevance, completeness, coherence,
and Turkish language quality -- without requiring an external LLM judge.
Each sub-score is normalized to the 0-1 range.
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Turkish-specific helpers
# ---------------------------------------------------------------------------

# Common Turkish suffixes (simplified) used to detect Turkish text.
_TURKISH_CHARS = set("cCgGiIoOsSuU")
_TURKISH_SPECIFIC_CHARS = set("\u00e7\u00c7\u011f\u011e\u0131\u0130\u00f6\u00d6\u015f\u015e\u00fc\u00dc")

# Typical Turkish stop-words that indicate genuine Turkish text.
_TURKISH_STOP_WORDS: set[str] = {
    "ve", "bir", "bu", "ile", "da", "de", "den", "dan", "icin",  # ASCII forms
    "i\u00e7in", "olan", "olarak", "gibi", "daha", "en", "ancak", "ama",
    "fakat", "hem", "ya", "veya", "ise", "kadar", "sonra", "\u00f6nce",
    "aras\u0131nda", "baz\u0131", "her", "\u00e7ok", "ne", "nas\u0131l",
    "neden", "hangi", "t\u00fcm", "b\u00fct\u00fcn",
}

# Structure markers that indicate a well-organized response.
_STRUCTURE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\s*\d+[\.\)]\s", re.MULTILINE),           # numbered list
    re.compile(r"^\s*[-*]\s", re.MULTILINE),                  # bullet list
    re.compile(r"^#{1,6}\s", re.MULTILINE),                   # markdown headings
    re.compile(r"\*\*.+?\*\*"),                                # bold emphasis
    re.compile(r"```[\s\S]+?```"),                             # code blocks
    re.compile(r"^\s*\w+:\s", re.MULTILINE),                  # key: value pairs
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class QualityScores:
    """Container for the four quality sub-scores and the aggregate."""

    relevance: float = 0.0
    completeness: float = 0.0
    coherence: float = 0.0
    turkish_quality: Optional[float] = None
    overall: float = 0.0

    def to_dict(self) -> dict[str, float | None]:
        return {
            "relevance": round(self.relevance, 4),
            "completeness": round(self.completeness, 4),
            "coherence": round(self.coherence, 4),
            "turkish_quality": round(self.turkish_quality, 4) if self.turkish_quality is not None else None,
            "overall": round(self.overall, 4),
        }


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def score_relevance(
    response: str,
    prompt: str,
    keywords: Optional[list[str]] = None,
) -> float:
    """Score how relevant the response is to the prompt.

    Heuristics:
    1. Keyword overlap between prompt and response (40%).
    2. Explicit keyword hit rate if *keywords* are provided (40%).
    3. Penalty for very short or obviously off-topic replies (20%).

    Returns a float in [0, 1].
    """
    if not response.strip():
        return 0.0

    response_lower = response.lower()
    prompt_lower = prompt.lower()

    # --- 1. Prompt-keyword overlap ---
    prompt_tokens = set(_tokenize(prompt_lower))
    response_tokens = set(_tokenize(response_lower))
    # Remove very short tokens (articles, prepositions) that add noise.
    prompt_tokens = {t for t in prompt_tokens if len(t) > 2}
    if prompt_tokens:
        overlap = len(prompt_tokens & response_tokens) / len(prompt_tokens)
    else:
        overlap = 0.5  # neutral when prompt has no meaningful tokens

    # --- 2. Explicit keyword hit rate ---
    if keywords:
        hits = sum(1 for kw in keywords if kw.lower() in response_lower)
        keyword_score = hits / len(keywords)
    else:
        keyword_score = overlap  # fall back to prompt overlap

    # --- 3. Length penalty ---
    word_count = len(response.split())
    if word_count < 5:
        length_factor = 0.2
    elif word_count < 20:
        length_factor = 0.6
    else:
        length_factor = 1.0

    score = 0.4 * overlap + 0.4 * keyword_score + 0.2 * length_factor
    return min(max(score, 0.0), 1.0)


def score_completeness(
    response: str,
    expected_min_words: int = 30,
    expected_max_words: int = 500,
) -> float:
    """Score how thorough / complete the response is.

    Heuristics:
    - Word count relative to the expected range (50%).
    - Presence of examples, explanations, or elaboration markers (30%).
    - Paragraph or section count (20%).

    Returns a float in [0, 1].
    """
    if not response.strip():
        return 0.0

    words = response.split()
    word_count = len(words)

    # --- Word count score (logistic-style) ---
    if word_count >= expected_min_words:
        # Gradually approach 1.0 as word count increases to expected_max.
        ratio = min(word_count / expected_max_words, 1.0)
        wc_score = 0.5 + 0.5 * ratio
    else:
        wc_score = 0.5 * (word_count / expected_min_words)

    # --- Elaboration markers ---
    elaboration_markers = [
        "for example", "such as", "e.g.", "because", "therefore",
        "however", "in addition", "furthermore", "specifically",
        "notably", "in particular", "this means",
        # Turkish equivalents
        "\u00f6rne\u011fin", "\u00e7\u00fcnk\u00fc", "dolay\u0131s\u0131yla",
        "ayr\u0131ca", "\u00f6zellikle", "bunun yan\u0131 s\u0131ra",
    ]
    response_lower = response.lower()
    marker_hits = sum(1 for m in elaboration_markers if m in response_lower)
    elab_score = min(marker_hits / 4.0, 1.0)

    # --- Section / paragraph count ---
    paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
    para_score = min(len(paragraphs) / 3.0, 1.0)

    score = 0.50 * wc_score + 0.30 * elab_score + 0.20 * para_score
    return min(max(score, 0.0), 1.0)


def score_coherence(response: str) -> float:
    """Score how well-structured and coherent the response is.

    Heuristics:
    - Sentence count and average sentence length (30%).
    - Structural formatting markers (lists, headings, bold) (40%).
    - Transition words indicating logical flow (30%).

    Returns a float in [0, 1].
    """
    if not response.strip():
        return 0.0

    # --- Sentence analysis ---
    sentences = re.split(r"[.!?]+", response)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = max(len(sentences), 1)
    avg_sentence_len = sum(len(s.split()) for s in sentences) / sentence_count

    # Penalize very short (<5 words) or very long (>40 words) average sentences.
    if 8 <= avg_sentence_len <= 30:
        sent_score = 1.0
    elif 5 <= avg_sentence_len < 8 or 30 < avg_sentence_len <= 40:
        sent_score = 0.7
    else:
        sent_score = 0.4

    # --- Structure markers ---
    structure_hits = sum(
        1 for pat in _STRUCTURE_PATTERNS if pat.search(response)
    )
    struct_score = min(structure_hits / 3.0, 1.0)

    # --- Transition words ---
    transitions = [
        "first", "second", "third", "finally", "next", "then",
        "moreover", "however", "additionally", "consequently",
        "in conclusion", "to summarize", "on the other hand",
        # Turkish
        "ilk olarak", "ikinci olarak", "son olarak",
        "bununla birlikte", "sonu\u00e7 olarak", "\u00f6zetle",
    ]
    response_lower = response.lower()
    trans_hits = sum(1 for t in transitions if t in response_lower)
    trans_score = min(trans_hits / 3.0, 1.0)

    score = 0.30 * sent_score + 0.40 * struct_score + 0.30 * trans_score
    return min(max(score, 0.0), 1.0)


def score_turkish_quality(response: str) -> float:
    """Score the quality of Turkish language in the response.

    Checks:
    - Presence of Turkish-specific characters (30%).
    - Turkish stop-word frequency (40%).
    - Absence of broken encoding artefacts (30%).

    Returns a float in [0, 1], or 0.0 if the response does not appear to
    be Turkish at all.
    """
    if not response.strip():
        return 0.0

    # --- Turkish character presence ---
    turkish_char_count = sum(1 for ch in response if ch in _TURKISH_SPECIFIC_CHARS)
    # Normalize: expect roughly 2-5% of characters to be Turkish-specific.
    char_ratio = turkish_char_count / max(len(response), 1)
    char_score = min(char_ratio / 0.03, 1.0)

    # --- Turkish stop-word frequency ---
    tokens = set(_tokenize(response.lower()))
    stop_hits = len(tokens & _TURKISH_STOP_WORDS)
    stop_score = min(stop_hits / 5.0, 1.0)

    # --- Encoding artefact check ---
    # Common signs of broken UTF-8: sequences like Ã¶, Ã¼, Ã§ etc.
    artefact_patterns = [r"\u00c3[\u00a4-\u00bc]", r"\ufffd", r"\\u00"]
    artefact_count = sum(
        len(re.findall(pat, response)) for pat in artefact_patterns
    )
    encoding_score = 1.0 if artefact_count == 0 else max(1.0 - artefact_count * 0.2, 0.0)

    # If virtually no Turkish detected, return 0.
    if char_score < 0.05 and stop_score < 0.1:
        return 0.0

    score = 0.30 * char_score + 0.40 * stop_score + 0.30 * encoding_score
    return min(max(score, 0.0), 1.0)


# ---------------------------------------------------------------------------
# Aggregate evaluator
# ---------------------------------------------------------------------------

def evaluate_response(
    response: str,
    prompt: str,
    keywords: Optional[list[str]] = None,
    is_turkish: bool = False,
    expected_min_words: int = 30,
    expected_max_words: int = 500,
) -> QualityScores:
    """Run all quality checks and return a :class:`QualityScores` instance.

    Parameters
    ----------
    response:
        The raw text returned by the model.
    prompt:
        The original prompt that was sent to the model.
    keywords:
        Optional list of keywords expected in a good answer.
    is_turkish:
        If ``True``, Turkish quality scoring is included in the overall score.
    expected_min_words / expected_max_words:
        Word-count range considered "complete" for the completeness metric.

    Returns
    -------
    QualityScores
        Dataclass with per-axis scores and an aggregate ``overall`` score.
    """
    rel = score_relevance(response, prompt, keywords)
    comp = score_completeness(response, expected_min_words, expected_max_words)
    coh = score_coherence(response)

    if is_turkish:
        turk = score_turkish_quality(response)
        # Weighted aggregate: relevance 30%, completeness 25%, coherence 25%, Turkish 20%.
        overall = 0.30 * rel + 0.25 * comp + 0.25 * coh + 0.20 * turk
    else:
        turk = None
        # Weighted aggregate without Turkish dimension.
        overall = 0.40 * rel + 0.30 * comp + 0.30 * coh

    scores = QualityScores(
        relevance=rel,
        completeness=comp,
        coherence=coh,
        turkish_quality=turk,
        overall=overall,
    )
    return scores


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Cheap whitespace + punctuation tokenizer."""
    return re.findall(r"\b\w+\b", text)
