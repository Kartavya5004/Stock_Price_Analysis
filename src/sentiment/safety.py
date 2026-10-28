"""Heuristic guard against instruction-like text hidden in headlines.

FinBERT is a classifier: it cannot *follow* instructions, but an attacker can still try to sway
the score, and any future LLM scorer would be directly exposed. Headlines that look like they
address the model are excluded from scoring (treated as untrusted data, not as news).

This is a keyword heuristic, not a guarantee: obfuscated phrasing can slip through. For an LLM
scorer also delimit the headline in the prompt, demand strict JSON, and clamp/validate the score.
"""
import re
import warnings

import pandas as pd

_PATTERNS = [
    r"\b(ignore|disregard|forget|override|bypass)\b\W+(?:\w+\W+){0,3}?(instructions?|prompts?)\b",
    r"\b(ignore|disregard|forget)\b\W+(?:all\W+|any\W+)?(previous|prior|above|earlier)\b\W+(messages?|context|text)\b",
    r"\b(system|developer)\s+(prompt|message|instructions?)\b",
    r"(?:^|[\n.:;]\s*)(system|assistant)\s*:",
    r"</?\s*(system|instructions?|prompt)\s*>",
    r"\b(score|rate|classify|label)\s+(this|it|the\s+(headline|article|text))\s+(as|at|to|with)\b",
    r"\b(respond|reply|output)\s+(only\s+)?with\b",
    r"\bnew\s+instructions?\b",
]
_RE = [re.compile(p, re.IGNORECASE) for p in _PATTERNS]


def is_injection(text: str) -> bool:
    return any(r.search(text or "") for r in _RE)


def split_flagged(news: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Return (news without flagged rows, number flagged). Needs a 'text' column."""
    if news.empty:
        return news, 0
    mask = news["text"].map(is_injection)
    n = int(mask.sum())
    if n:
        warnings.warn(f"Excluded {n} headline(s) containing instruction-like text from sentiment scoring.")
    return news[~mask], n
