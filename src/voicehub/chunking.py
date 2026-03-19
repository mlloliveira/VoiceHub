from __future__ import annotations

import re
from functools import lru_cache
from typing import List, Sequence, Tuple

try:
    import spacy
except Exception:  # pragma: no cover - optional dependency at runtime
    spacy = None

# Legacy regex kept as a backup path.
_SENT_BOUNDARY_RE = re.compile(r'(?<=[\.\!\?\:;\u3002])\s+')
_WORDLIKE_RE = re.compile(r"\w+", flags=re.UNICODE)


def _normalize_lang_code(lang_code: str | None) -> str:
    code = (lang_code or "xx").strip().lower()
    if not code:
        return "xx"
    if code.startswith("zh"):
        return "zh"
    return code.split("-")[0]


@lru_cache(maxsize=8)
def _get_sentencizer(lang_code: str):
    if spacy is None:
        return None

    normalized = _normalize_lang_code(lang_code)
    candidates = [normalized, "xx"] if normalized != "xx" else ["xx"]
    for candidate in candidates:
        try:
            nlp = spacy.blank(candidate)
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            return nlp
        except Exception:
            continue
    return None


def stable_wordlike_sequence(text: str) -> List[str]:
    return _WORDLIKE_RE.findall(text or "")


def content_preserved(original: str, candidate: str) -> bool:
    return stable_wordlike_sequence(original) == stable_wordlike_sequence(candidate)


def choose_safe_refined_text(original: str, candidate: str) -> Tuple[str, bool]:
    """
    Keep Ollama's output only when the word-like token sequence is preserved.
    This rejects dropped, reordered, or mutated IDs/text while still allowing
    punctuation-only cleanup.
    """
    original = original or ""
    candidate = candidate or ""
    if not candidate:
        return original, False
    if content_preserved(original, candidate):
        return candidate, True
    return original, False


def _legacy_sentence_segments(text: str) -> List[str]:
    if not text:
        return []

    out: List[str] = []
    last = 0
    for match in _SENT_BOUNDARY_RE.finditer(text):
        end = match.start()
        seg = text[last:end]
        if seg:
            out.append(seg)
        last = end
    tail = text[last:]
    if tail:
        out.append(tail)
    return [seg for seg in out if seg]


def _library_sentence_segments(text: str, lang_code: str = "xx") -> List[str]:
    nlp = _get_sentencizer(lang_code)
    if nlp is None or not text:
        return []

    doc = nlp(text)
    sents = list(doc.sents)
    if not sents:
        return []

    segments = [sent.text_with_ws for sent in sents if sent.text_with_ws]
    if "".join(segments) == text:
        return segments

    # Fallback to explicit character spans if text_with_ws did not reconstruct exactly.
    spans: List[str] = []
    for idx, sent in enumerate(sents):
        start = sent.start_char
        end = sents[idx + 1].start_char if idx + 1 < len(sents) else len(text)
        seg = text[start:end]
        if seg:
            spans.append(seg)
    if "".join(spans) == text:
        return spans
    return []


def _split_oversized_segment_exact(segment: str, max_len: int) -> List[str]:
    if len(segment) <= max_len:
        return [segment]

    pieces: List[str] = []
    remaining = segment
    while len(remaining) > max_len:
        window = remaining[:max_len]
        break_at = None

        # Prefer a boundary at trailing whitespace within the allowed window.
        for match in re.finditer(r"\s+", window):
            break_at = match.end()

        # Otherwise prefer the latest sentence-ish punctuation inside the window.
        if break_at is None:
            punct_positions = [
                pos + 1 for pos, ch in enumerate(window)
                if ch in ".!?;:\u3002"
            ]
            if punct_positions:
                break_at = punct_positions[-1]

        if break_at is None or break_at <= 0:
            break_at = max_len

        pieces.append(remaining[:break_at])
        remaining = remaining[break_at:]

    if remaining:
        pieces.append(remaining)
    return [piece for piece in pieces if piece]


def _pack_segments_exact(segments: Sequence[str], max_len: int) -> List[str]:
    chunks: List[str] = []
    current = ""

    for segment in segments:
        if not segment:
            continue

        if len(segment) <= max_len:
            if not current:
                current = segment
            elif len(current) + len(segment) <= max_len:
                current += segment
            else:
                chunks.append(current)
                current = segment
            continue

        if current:
            chunks.append(current)
            current = ""

        chunks.extend(_split_oversized_segment_exact(segment, max_len))

    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk]


def _legacy_chunk_text(text: str, max_len: int) -> Tuple[List[str], List[str], str]:
    sentences = _legacy_sentence_segments(text)
    if not sentences:
        return [], [], "legacy-regex"
    chunks = _pack_segments_exact(sentences, max_len)
    return sentences, chunks, "legacy-regex"


def chunk_text(text: str, max_len: int, lang_code: str = "xx") -> Tuple[List[str], List[str], str]:
    """
    Primary chunker: spaCy sentencizer + exact-text packing.
    Backup chunker: legacy regex splitter.

    Returned chunks are exact substrings whose concatenation reproduces the input
    text byte-for-byte (for normal Python string semantics).
    """
    text = text or ""
    max_len = max(1, int(max_len))
    if not text:
        return [], [], "empty"

    sentences = _library_sentence_segments(text, lang_code=lang_code)
    if sentences:
        chunks = _pack_segments_exact(sentences, max_len)
        if chunks and "".join(chunks) == text and all(len(chunk) <= max_len for chunk in chunks):
            return sentences, chunks, "spacy-sentencizer"

    return _legacy_chunk_text(text, max_len)
