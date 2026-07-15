"""Stage 2 — validate raw teacher output and write the training-ready CSV.

Usage:  uv run python -m sdg.validate
Reads  data/raw/synthetic_slang.raw.jsonl
Writes data/raw/synthetic_slang.csv  (schema matches an existing SOURCES entry)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter

from config import EVAL_PATH, RAW_DIR, SDG_PATH
from prepare_data import banned_from_eval, load_existing_eval

RAW_IN = RAW_DIR / "synthetic_slang.raw.jsonl"
_PREAMBLE = ("sure, here", "here is", "here's", "certainly")


def check_row(row: dict, banned: set) -> str | None:
    slang = str(row.get("slang", "")).strip()
    english = str(row.get("english", "")).strip()
    if not slang or not english:
        return "empty-field"
    if not (3 <= len(slang) <= 200) or not (3 <= len(english) <= 200):
        return "bad-length"
    low_s = slang.lower()
    if any(low_s.startswith(p) for p in _PREAMBLE):
        return "preamble-leak"
    if low_s in banned or english.lower() in banned:
        return "eval-leak"
    if not row.get("is_hard_negative"):
        term = str(row.get("term", "")).strip().lower()
        if term and term not in low_s:
            return "term-missing"
    return None


def dedupe_rows(rows: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for r in rows:
        key = (str(r.get("slang", "")).lower(), str(r.get("english", "")).lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def main(argv=None) -> int:
    argparse.ArgumentParser().parse_args(argv)  # no flags; keeps CLI shape consistent
    if not RAW_IN.exists():
        print(f"No raw file at {RAW_IN}. Run  uv run python -m sdg.generate  first.")
        return 1

    eval_items = load_existing_eval() or []
    if not eval_items:
        print("!! WARNING: no frozen eval.jsonl found; leak guard is empty.")
    banned = banned_from_eval(eval_items)

    rows = [json.loads(l) for l in open(RAW_IN, encoding="utf-8")]
    kept, reasons = [], Counter()
    for r in rows:
        why = check_row(r, banned)
        if why:
            reasons[why] += 1
        else:
            kept.append(r)
    kept = dedupe_rows(kept)

    with open(SDG_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["slang_sentence", "normal_sentence", "slang_term",
                    "tone", "difficulty", "is_hard_negative"])
        for r in kept:
            w.writerow([r["slang"], r["english"], r.get("term", ""),
                        r.get("tone", ""), r.get("difficulty", ""),
                        bool(r.get("is_hard_negative"))])

    print(f"generated {len(rows)} -> kept {len(kept)}")
    print(f"rejected by reason: {dict(reasons)}")
    print(f"Wrote {SDG_PATH}. Next: register it in SOURCES (Task 6).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
