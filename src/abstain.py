"""Abstention: recognise unclear input and decline instead of hallucinating.

Two layers (see design):
  1. A cheap INPUT GUARD (`looks_unclear`) used by the app RIGHT NOW so blatant
     junk (empty, symbols-only, too short) gets an honest "I'm not sure" without
     even calling the model.
  2. TRAINING EXAMPLES + EVAL ITEMS so a future retrain teaches the model to
     abstain on subtler unclear inputs, and so we can MEASURE abstention.

The eval deliberately does NOT use the guard (it measures the raw model), so the
before/after honestly shows whether training improved abstention.
"""

from __future__ import annotations

import random
import string

from config import (
    ABSTAIN_MARKERS,
    ABSTAIN_MESSAGE,
    TAG_TO_ENGLISH,
    TAG_TO_SLANG,
)


def looks_unclear(text: str) -> bool:
    """Cheap, CONSERVATIVE guard: only True for input that clearly isn't
    translatable text. Deliberately does NOT try to catch subtle gibberish
    (that's the trained model's job) so it never mis-fires on real slang
    like 'fr', 'rizz', 'delulu'.
    """
    t = (text or "").strip()
    if len(t) < 2:
        return True                       # empty or a single character
    if not any(c.isalpha() for c in t):
        return True                       # only digits / punctuation / emoji
    return False


def is_abstention(output: str) -> bool:
    """True if a model output reads as an abstention (contains a marker)."""
    o = (output or "").lower()
    return any(m in o for m in ABSTAIN_MARKERS)


# ---------------------------------------------------------------------------
# Curated unanswerable inputs for the frozen EVAL set. Clearly not translatable
# slang/English -> the right behaviour is to abstain.
# ---------------------------------------------------------------------------
UNANSWERABLE_EVAL_INPUTS = [
    "asdfghjkl qwerty zxcvbnm",
    "########## $$$$$ %%%%",
    "42 42 42 42 42",
    "zxqvwk bnmpld",
    "lorem ipsum dolor sit amet",
    "aaaaaaaaaaaaaaaa",
    "?!?!?! ...... ?!?!",
    "qwertyuiop asdfghjkl",
    "blorptangle frimbulate",
    "xkcd plmokn wsxedc",
]


def make_abstain_eval_items() -> list[dict]:
    """Eval items whose correct behaviour is to abstain."""
    items = []
    for i, inp in enumerate(UNANSWERABLE_EVAL_INPUTS):
        items.append({
            "id": f"un_{i:03d}",
            "direction": "unanswerable",
            "type": "unanswerable",
            "tag": TAG_TO_ENGLISH,
            "input": inp,
            "reference": ABSTAIN_MESSAGE,   # expected: abstain
            "term": "",
            "meaning": "(unanswerable — the model should abstain)",
            "strat": "unanswerable",
        })
    return items


def _random_unclear(rng: random.Random) -> str:
    """Generate one 'unclear' input for a TRAINING abstain example."""
    kind = rng.choice(["mash", "symbols", "repeat_word", "repeat_char", "letters"])
    if kind == "mash":
        n = rng.randint(2, 4)
        toks = ["".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(5, 10)))
                for _ in range(n)]
        return " ".join(toks)
    if kind == "symbols":
        return "".join(rng.choice("!@#$%^&*()_+-=[]{};:,.<>?/") for _ in range(rng.randint(4, 12)))
    if kind == "repeat_word":
        w = "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(3, 6)))
        return " ".join([w] * rng.randint(3, 6))
    if kind == "repeat_char":
        return rng.choice(string.ascii_lowercase) * rng.randint(8, 16)
    # single nonsense letters
    return " ".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(4, 8)))


def make_abstain_train_examples(rng: random.Random, n: int) -> list[dict]:
    """Synthetic 'unclear input -> abstain' training examples, both directions."""
    exs = []
    for _ in range(n):
        inp = _random_unclear(rng)
        tag = rng.choice([TAG_TO_ENGLISH, TAG_TO_SLANG])
        exs.append({
            "messages": [
                {"role": "user", "content": f"{tag}\n{inp}"},
                {"role": "assistant", "content": ABSTAIN_MESSAGE},
            ]
        })
    return exs
