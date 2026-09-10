"""Remove shared company boilerplate from a job description before embedding.

The relevance bug this fixes: many postings from the same company open with an
identical "About <company>" preamble (LangChain's runs ~205 words / 1,356 chars).
``all-MiniLM-L6-v2`` truncates at ~256 word-pieces, so the boilerplate consumes
the whole embedding window and the role-specific text never reaches the model —
every role at that company embeds to nearly the same vector, and Stage A can no
longer tell a FullStack role from a Customer Engineer role. The same preamble
also dominated the Phase-2 requirement extractor (empty ``required[]`` for jobs
whose distinguishing text was truncated away).

Fix (deterministic, no LLM): within a set of same-company postings, drop the
paragraph BLOCKS that recur across postings (boilerplate) and keep the blocks
unique to each posting (the actual role). A strict common-prefix is too fragile
here — LangChain has 40 distinct 1,356-char prefixes across 95 postings — so we
compare at the paragraph level and by document frequency, which also catches
boilerplate in the middle or tail (benefits, EEO statements), not just a prefix.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

_BLOCK_SPLIT = re.compile(r"\n\s*\n+")  # blank-line-separated paragraphs
_WS = re.compile(r"\s+")


def _blocks(text: str) -> list[str]:
    return [b.strip() for b in _BLOCK_SPLIT.split(text) if b.strip()]


def _norm(block: str) -> str:
    """Whitespace- and case-insensitive key for comparing blocks across postings."""
    return _WS.sub(" ", block).strip().casefold()


def strip_shared_blocks(
    texts: list[str], *, min_block_chars: int = 40, min_shared: int = 2
) -> list[str]:
    """Return ``texts`` with same-company boilerplate paragraph blocks removed.

    A block is boilerplate if its normalized form appears in at least
    ``min_shared`` DISTINCT postings and is at least ``min_block_chars`` long.
    Short shared blocks (section headers like "Responsibilities:") stay, so a
    posting never collapses to nothing over headers alone. With fewer than two
    texts there is nothing to compare and inputs are returned unchanged. If
    stripping would empty a posting, that posting is returned unchanged — a
    fail-safe so we never hand an empty string to the embedder.
    """
    if len(texts) < 2:
        return list(texts)
    doc_freq: Counter[str] = Counter()
    per_text_blocks: list[list[str]] = []
    for t in texts:
        bl = _blocks(t)
        per_text_blocks.append(bl)
        for norm in {_norm(b) for b in bl}:  # once per posting (document freq)
            doc_freq[norm] += 1
    shared = {
        norm
        for norm, df in doc_freq.items()
        if df >= min_shared and len(norm) >= min_block_chars
    }
    out: list[str] = []
    for original, bl in zip(texts, per_text_blocks, strict=True):
        kept = [b for b in bl if _norm(b) not in shared]
        cleaned = "\n\n".join(kept).strip()
        out.append(cleaned or original)
    return out


def deboilerplate_by_company(
    items: list[tuple[str, str, str]],
) -> dict[str, str]:
    """De-boilerplate a mixed set of postings, grouping by company first.

    ``items`` are ``(job_id, company, description_text)``; returns
    ``{job_id: cleaned_text}``. Postings are compared only within their own
    company, so one company's boilerplate never strips another's content.
    """
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for jid, company, text in items:
        groups[company].append((jid, text))
    out: dict[str, str] = {}
    for members in groups.values():
        cleaned = strip_shared_blocks([t for _, t in members])
        for (jid, _), c in zip(members, cleaned, strict=True):
            out[jid] = c
    return out
