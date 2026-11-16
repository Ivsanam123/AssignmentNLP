"""Abbreviation detection for BC5CDR sentences (text only, no labels needed).

Implements the Schwartz & Hearst (2003) long-form / short-form matching on the "long form ( SHORT )" pattern,
adapted to the dataset's token lists (brackets are separate tokens, e.g. "indomethacin ( IDM )").
Only the "long form ( SHORT )" order is handled; "SHORT ( long form )" is not.
"""
import re


def _match_long_form(short, long_tokens):
    """S&H backward character matching. Returns (start_index_in_long_tokens, ) of the best long form or None."""
    s = short.lower()
    long_str = " ".join(long_tokens).lower()
    # char positions -> which token each char belongs to
    s_idx = len(s) - 1
    l_idx = len(long_str) - 1
    while s_idx >= 0:
        c = s[s_idx]
        if not c.isalnum():
            s_idx -= 1
            continue
        while l_idx >= 0 and (long_str[l_idx] != c or
                              (s_idx == 0 and l_idx > 0 and long_str[l_idx - 1].isalnum())):
            l_idx -= 1
        if l_idx < 0:
            return None
        l_idx -= 1
        s_idx -= 1
    # l_idx is now just before the start char; the long form starts at the start of the word containing l_idx+1
    start_char = l_idx + 1
    while start_char > 0 and long_str[start_char - 1] != " ":
        start_char -= 1
    n_tokens_before = long_str[:start_char].count(" ")
    return n_tokens_before


def find_pairs(tokens):
    """Return list of dicts {short, short_pos, long_start, long_end} for each valid definition in the sentence."""
    out = []
    n = len(tokens)
    for i, t in enumerate(tokens):
        if t != "(":
            continue
        # short form = 1 or 2 tokens before the closing bracket
        for j in range(i + 2, min(i + 4, n)):
            if tokens[j] != ")":
                continue
            short_toks = tokens[i + 1:j]
            short = " ".join(short_toks)
            letters = sum(c.isalpha() for c in short)
            if not (2 <= len(short) <= 10) or letters == 0 or not short[0].isalnum():
                break
            if len(short_toks) > 1 and not all(w.isupper() or w.isdigit() for w in short_toks):
                break
            a = sum(c.isalnum() for c in short)
            max_words = min(a + 5, a * 2)
            lo = max(0, i - max_words)
            cand = tokens[lo:i]
            k = _match_long_form(short, cand)
            if k is None:
                break
            ls = lo + k
            if i - ls < 1 or len(" ".join(tokens[ls:i])) <= len(short):
                break
            # long form must not itself contain brackets
            if any(w in "()[]" for w in tokens[ls:i]):
                break
            out.append({"short": short, "short_pos": (i + 1, j), "long_start": ls, "long_end": i,
                        "long": " ".join(tokens[ls:i])})
            break
    return out


def spans(tags):
    """BIO tag list -> set of (start, end_exclusive, type); a stray I- starts a span."""
    out, start, typ = set(), None, None
    for i, t in enumerate(list(tags) + ["O"]):
        if t.startswith("B-") or t == "O" or (t.startswith("I-") and t[2:] != typ):
            if start is not None:
                out.add((start, i, typ))
                start, typ = None, None
            if t.startswith("B-") or t.startswith("I-"):
                start, typ = i, t[2:]
    return out
