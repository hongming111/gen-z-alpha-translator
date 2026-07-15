# NVIDIA Pipeline — Phase 1 (Eval + SDG + Retrain) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared NVIDIA-API teacher client, the ICL ceiling test (Stage 1), and the Constraint-First synthetic-data generator (Stage 2), then fold the synthetic data into the existing training pipeline for a retrain (Stage 3).

**Architecture:** New stage modules under `src/` bolt onto the existing `config.py` + `prepare_data.py` drop-in data pipeline. A shared `src/teacher.py` wraps the NVIDIA API (OpenAI-compatible) for both generation and LLM-as-judge. Stage 2 samples example *attributes* first (Constraint-First), then asks the teacher to write a slang↔English pair fitting them; validated output lands in `data/raw/synthetic_slang.csv` and is registered in `SOURCES` so `prepare_data.py` folds it in automatically. Pure-logic units are TDD'd with pytest; API/GPU steps are verified by pilot/smoke runs with manual inspection (NVIDIA's "pilot before scale").

**Tech Stack:** Python 3.11, uv, pandas, `openai` (NVIDIA endpoint), `python-dotenv`, pytest. Existing: unsloth/TRL for training (unchanged this phase).

## Global Constraints

- Python pinned `==3.11.*`; all commands run via `uv run`.
- Never commit secrets: `NVIDIA_API_KEY` lives only in a gitignored `.env`. Never print the key; never accept it via chat.
- The frozen eval set `data/processed/eval.jsonl` (70 items) MUST NOT change and MUST NOT leak into any training/synthetic data.
- Deterministic where possible: fixed seed `RANDOM_SEED = 42` (already in `config.py`); reuse it for the SDG sampler.
- Teacher/judge model id is a single config constant (`TEACHER_MODEL`) so it can be swapped if unavailable on the user's account.
- Direction tags are the existing constants `TAG_TO_ENGLISH = "Translate to English:"` and `TAG_TO_SLANG = "Translate to Gen Z slang:"` — never re-typed as literals.
- Synthetic CSV schema matches the existing augmented source so `SOURCES` can consume it: columns `slang_sentence`, `normal_sentence` (+ metadata columns ignored by the loader).

---

### Task 1: Project setup — deps, secrets, config, teacher client

**Files:**
- Modify: `pyproject.toml` (add deps)
- Modify: `.gitignore` (add `.env`)
- Create: `.env.example`
- Modify: `src/config.py` (append SDG/teacher config block)
- Create: `src/teacher.py`
- Create: `tests/__init__.py` (empty)
- Test: `tests/test_teacher.py`

**Interfaces:**
- Produces:
  - `src/config.py` constants: `NVIDIA_BASE_URL: str`, `TEACHER_MODEL: str`, `JUDGE_MODEL: str`, `SDG_PATH: Path` (= `RAW_DIR/"synthetic_slang.csv"`), `SDG_TARGET: int`, `SDG_HARD_NEG_FRAC: float`, `SDG_DIRECTION_WEIGHTS: list[tuple[str,float]]`, `SDG_TONES: list[str]`, `SDG_CONTEXTS: list[str]`, `SDG_DIFFICULTY_WEIGHTS: list[tuple[str,float]]`.
  - `src/teacher.py`:
    - `get_client() -> openai.OpenAI` — builds a client from `NVIDIA_API_KEY`; raises `RuntimeError("Set NVIDIA_API_KEY in .env")` if missing.
    - `chat(client, prompt: str, *, system: str = "", temperature: float = 0.7, max_tokens: int = 512) -> str` — one completion; returns text content.
    - `extract_json(text: str) -> dict | None` — returns the first balanced `{...}` block that parses as JSON (stripping `<think>...</think>` wrappers), else `None`.

- [ ] **Step 1: Add dependencies**

Edit `pyproject.toml` — add to `dependencies`:
```toml
    "openai>=1.40",        # NVIDIA API is OpenAI-compatible (teacher + judge)
    "python-dotenv>=1.0",  # load NVIDIA_API_KEY from a gitignored .env
```
Add a new dev group at the end of `[project.optional-dependencies]`:
```toml
dev = [
    "pytest>=8.0",
]
```

- [ ] **Step 2: Install**

Run: `uv sync --extra dev`
Expected: resolves and installs `openai`, `python-dotenv`, `pytest` (no errors).

- [ ] **Step 3: Gitignore the secret + add an example**

Append to `.gitignore`:
```
# Secrets
.env
```
Create `.env.example`:
```
# Copy to .env and paste your free key from https://build.nvidia.com
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

- [ ] **Step 4: Append SDG/teacher config to `src/config.py`**

Append at the end of `src/config.py`:
```python
# ---------------------------------------------------------------------------
# NVIDIA teacher/judge (SDG stage 2 + DPO stage 4). Key comes from .env
# (NVIDIA_API_KEY) — never hard-code it. Model id is one constant so it can be
# swapped if a given model isn't enabled on your build.nvidia.com account.
# ---------------------------------------------------------------------------
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
TEACHER_MODEL = "nvidia/llama-3.1-nemotron-70b-instruct"
JUDGE_MODEL = TEACHER_MODEL

# Constraint-First SDG settings.
SDG_PATH = RAW_DIR / "synthetic_slang.csv"
SDG_TARGET = 1200                       # kept pairs to aim for
SDG_HARD_NEG_FRAC = 0.12                # fraction generated as "bait" hard negatives
SDG_DIRECTION_WEIGHTS = [("to_english", 0.65), ("to_slang", 0.35)]  # aim at weak dir
SDG_DIFFICULTY_WEIGHTS = [("clear", 0.5), ("ambiguous", 0.3), ("edge", 0.2)]
SDG_TONES = ["playful", "hype", "sarcastic", "deadpan", "annoyed",
             "affectionate", "dramatic", "chill"]
SDG_CONTEXTS = ["texting a friend", "group chat", "gaming voice chat",
                "social media caption", "replying to a post", "DM to a crush"]
```

- [ ] **Step 5: Write the failing test for `extract_json`**

Create `tests/__init__.py` (empty) and `tests/test_teacher.py`:
```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from teacher import extract_json


def test_extract_json_plain():
    assert extract_json('{"slang": "delulu", "english": "delusional"}') == {
        "slang": "delulu", "english": "delusional"}


def test_extract_json_with_prose_around():
    text = 'Sure! Here is the pair:\n{"slang": "rizz", "english": "charm"}\nHope that helps.'
    assert extract_json(text) == {"slang": "rizz", "english": "charm"}


def test_extract_json_strips_think_wrapper():
    text = '<think>let me reason</think>{"a": 1}'
    assert extract_json(text) == {"a": 1}


def test_extract_json_none_on_garbage():
    assert extract_json("no json here") is None
```

- [ ] **Step 6: Run the test to verify it fails**

Run: `uv run pytest tests/test_teacher.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'teacher'` (file not created yet).

- [ ] **Step 7: Implement `src/teacher.py`**

Create `src/teacher.py`:
```python
"""Shared NVIDIA API client (OpenAI-compatible) for SDG + judging.

Key is read from a gitignored .env (NVIDIA_API_KEY). Get a free key at
https://build.nvidia.com. Model id is config.TEACHER_MODEL.
"""
from __future__ import annotations

import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

from config import NVIDIA_BASE_URL

load_dotenv()  # read .env into os.environ if present

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def get_client() -> OpenAI:
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        raise RuntimeError("Set NVIDIA_API_KEY in .env (see .env.example).")
    return OpenAI(base_url=NVIDIA_BASE_URL, api_key=key)


def chat(client: OpenAI, prompt: str, *, system: str = "",
         temperature: float = 0.7, max_tokens: int = 512) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=os.environ.get("TEACHER_MODEL_OVERRIDE") or _default_model(),
        messages=messages, temperature=temperature, max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def _default_model() -> str:
    from config import TEACHER_MODEL
    return TEACHER_MODEL


def extract_json(text: str) -> dict | None:
    """First balanced {...} block that parses as JSON, else None."""
    text = _THINK.sub("", text or "")
    for start in range(len(text)):
        if text[start] != "{":
            continue
        depth = 0
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:end + 1])
                    except json.JSONDecodeError:
                        break
    return None
```

- [ ] **Step 8: Run the test to verify it passes**

Run: `uv run pytest tests/test_teacher.py -v`
Expected: PASS (4 passed).

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml uv.lock .gitignore .env.example src/config.py src/teacher.py tests/
git commit -m "feat: NVIDIA teacher client + SDG config + secrets setup"
```

---

### Task 2: Stage 1 — ICL ceiling test (`src/eval_icl.py`)

**Files:**
- Create: `src/eval_icl.py`
- Test: `tests/test_eval_icl.py`

**Interfaces:**
- Consumes: `src/config.py` (`DICT_SOURCES`, `DICT_DIR`, `EVAL_PATH`, tags); `src/teacher.py` (`get_client`, `chat`, `extract_json`).
- Produces:
  - `build_glossary(max_terms: int = 40) -> str` — a `"term = meaning"` block (newline-joined) built from the dictionary CSVs, capped at `max_terms`.
  - `judge_translation(client, direction: str, source: str, reference: str, candidate: str) -> int` — returns `1` if the judge rules the candidate correct in meaning, else `0`.
  - `main() -> int` — runs the base model over `eval.jsonl` with and without the glossary, prints a per-direction before/after table. (CLI entry.)

- [ ] **Step 1: Write the failing test for `build_glossary`**

Create `tests/test_eval_icl.py`:
```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eval_icl import build_glossary


def test_glossary_caps_and_formats():
    g = build_glossary(max_terms=5)
    lines = [ln for ln in g.splitlines() if ln.strip()]
    assert len(lines) <= 5
    assert all(" = " in ln for ln in lines)  # "term = meaning"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_eval_icl.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval_icl'`.

- [ ] **Step 3: Implement `src/eval_icl.py`**

Create `src/eval_icl.py`:
```python
"""Stage 1 — ICL ceiling test.

Runs the BASE model over the frozen eval set twice: plain, and with a slang
glossary injected into the prompt. Scores per-direction with the NVIDIA judge.
If the glossary fixes slang->English, prompting is enough; if not, we've proven
that direction needs training data (justifies Stage 2 SDG).

Usage:  uv run python src/eval_icl.py
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from config import (DICT_DIR, DICT_SOURCES, EVAL_PATH, JUDGE_MODEL,
                    TAG_TO_ENGLISH, TAG_TO_SLANG)
from teacher import chat, extract_json, get_client


def build_glossary(max_terms: int = 40) -> str:
    terms: list[tuple[str, str]] = []
    seen = set()
    for src in DICT_SOURCES:
        if src.get("emoji"):
            continue
        p = DICT_DIR / src["file"]
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if src["term_col"] not in df.columns or src["meaning_col"] not in df.columns:
            continue
        for _, row in df.iterrows():
            t = str(row.get(src["term_col"], "")).strip()
            m = str(row.get(src["meaning_col"], "")).strip()
            if not t or not m or t.lower() in seen:
                continue
            seen.add(t.lower())
            terms.append((t, m))
            if len(terms) >= max_terms:
                return "\n".join(f"{t} = {m}" for t, m in terms)
    return "\n".join(f"{t} = {m}" for t, m in terms)


def judge_translation(client, direction: str, source: str, reference: str,
                      candidate: str) -> int:
    lang = "plain English" if direction == "to_english" else "Gen Z slang"
    prompt = (
        f"You are grading a translation into {lang}.\n"
        f"Source: {source}\nReference answer: {reference}\n"
        f"Candidate answer: {candidate}\n"
        "Is the candidate correct IN MEANING (wording may differ)? "
        "Reply with ONLY a JSON object: {\"correct\": true} or {\"correct\": false}."
    )
    out = chat(client, prompt, temperature=0.0, max_tokens=200)
    j = extract_json(out) or {}
    return 1 if j.get("correct") is True else 0


def _base_translate(client_unused, model, tokenizer, tag: str, text: str,
                    glossary: str) -> str:
    from translate_core import generate_translation
    prompt_text = text if not glossary else f"Glossary:\n{glossary}\n\n{text}"
    return generate_translation(model, tokenizer, tag, prompt_text)


def main() -> int:
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    eval_rows = [json.loads(l) for l in open(EVAL_PATH, encoding="utf-8")]
    translate_rows = [r for r in eval_rows if r.get("type") == "translate"]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        max_seq_length=1024, dtype=None, load_in_4bit=True)
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    FastLanguageModel.for_inference(model)

    glossary = build_glossary()
    client = get_client()

    results = {("plain", "to_english"): [], ("plain", "to_slang"): [],
               ("glossary", "to_english"): [], ("glossary", "to_slang"): []}
    for cond, gloss in [("plain", ""), ("glossary", glossary)]:
        for r in translate_rows:
            cand = _base_translate(None, model, tokenizer, r["tag"], r["input"], gloss)
            score = judge_translation(client, r["direction"], r["input"],
                                      r["reference"], cand)
            results[(cond, r["direction"])].append(score)

    print("\n=== ICL CEILING TEST (base model, judged per direction) ===")
    for direction in ("to_english", "to_slang"):
        p = results[("plain", direction)]
        g = results[("glossary", direction)]
        pa = sum(p) / len(p) if p else 0
        ga = sum(g) / len(g) if g else 0
        print(f"{direction:>11}: plain {pa:.0%}  ->  +glossary {ga:.0%}  (n={len(p)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_eval_icl.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Manual smoke run (GATED on NVIDIA_API_KEY + GPU)**

Precondition: `.env` has a valid `NVIDIA_API_KEY`, and no other process is holding GPU VRAM.
Run: `uv run python src/eval_icl.py`
Expected: prints a per-direction table like `to_english: plain 40% -> +glossary 55%`. Read it: record whether the glossary closes the slang→English gap. (This number goes on the "Gap" slide.)

- [ ] **Step 6: Commit**

```bash
git add src/eval_icl.py tests/test_eval_icl.py
git commit -m "feat: Stage 1 ICL ceiling test (glossary-in-prompt vs base)"
```

---

### Task 3: Stage 2 — Constraint-First attribute sampler (`src/sdg/attributes.py`)

**Files:**
- Create: `src/sdg/__init__.py` (empty)
- Create: `src/sdg/attributes.py`
- Test: `tests/test_attributes.py`

**Interfaces:**
- Consumes: `config` (`DICT_DIR`, `DICT_SOURCES`, `RANDOM_SEED`, `SDG_*` weight/pool constants).
- Produces:
  - `load_term_pool() -> list[str]` — unique slang terms from the dictionary CSVs.
  - `Recipe` dataclass: fields `direction: str`, `term: str`, `tone: str`, `difficulty: str`, `context: str`, `is_hard_negative: bool`.
  - `sample_recipes(n: int, seed: int = RANDOM_SEED) -> list[Recipe]` — deterministic list of `n` recipes honoring the configured weights and `SDG_HARD_NEG_FRAC`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_attributes.py`:
```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sdg.attributes import Recipe, sample_recipes


def test_sample_is_deterministic():
    a = sample_recipes(50, seed=42)
    b = sample_recipes(50, seed=42)
    assert [r.__dict__ for r in a] == [r.__dict__ for r in b]


def test_sample_respects_count_and_fields():
    recs = sample_recipes(30, seed=1)
    assert len(recs) == 30
    assert all(isinstance(r, Recipe) for r in recs)
    assert all(r.direction in ("to_english", "to_slang") for r in recs)
    assert all(r.term for r in recs)


def test_direction_weighting_favors_to_english():
    recs = sample_recipes(400, seed=7)
    to_eng = sum(1 for r in recs if r.direction == "to_english")
    assert to_eng > len(recs) * 0.5  # weighted 0.65
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_attributes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sdg'`.

- [ ] **Step 3: Implement `src/sdg/attributes.py`**

Create `src/sdg/__init__.py` (empty) and `src/sdg/attributes.py`:
```python
"""Constraint-First: sample the example RECIPE (attributes) before any teacher
call. Labels come from here (the spec), so the teacher can't mislabel them.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import pandas as pd

from config import (DICT_DIR, DICT_SOURCES, RANDOM_SEED, SDG_CONTEXTS,
                    SDG_DIFFICULTY_WEIGHTS, SDG_DIRECTION_WEIGHTS,
                    SDG_HARD_NEG_FRAC, SDG_TONES)


@dataclass
class Recipe:
    direction: str
    term: str
    tone: str
    difficulty: str
    context: str
    is_hard_negative: bool


def load_term_pool() -> list[str]:
    terms: set[str] = set()
    for src in DICT_SOURCES:
        if src.get("emoji"):
            continue
        p = DICT_DIR / src["file"]
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if src["term_col"] in df.columns:
            terms |= {str(t).strip() for t in df[src["term_col"]].dropna()}
    return sorted({t for t in terms if t and 1 <= len(t) <= 40})


def _weighted(rng: random.Random, pairs: list[tuple[str, float]]) -> str:
    vals = [v for v, _ in pairs]
    weights = [w for _, w in pairs]
    return rng.choices(vals, weights=weights, k=1)[0]


def sample_recipes(n: int, seed: int = RANDOM_SEED) -> list[Recipe]:
    rng = random.Random(seed)
    pool = load_term_pool()
    if not pool:
        raise RuntimeError("Empty term pool — check DICT_SOURCES / data/dictionaries.")
    out: list[Recipe] = []
    for _ in range(n):
        out.append(Recipe(
            direction=_weighted(rng, SDG_DIRECTION_WEIGHTS),
            term=rng.choice(pool),
            tone=rng.choice(SDG_TONES),
            difficulty=_weighted(rng, SDG_DIFFICULTY_WEIGHTS),
            context=rng.choice(SDG_CONTEXTS),
            is_hard_negative=(rng.random() < SDG_HARD_NEG_FRAC),
        ))
    return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_attributes.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/sdg/__init__.py src/sdg/attributes.py tests/test_attributes.py
git commit -m "feat: Stage 2 Constraint-First attribute sampler"
```

---

### Task 4: Stage 2 — teacher prompt builder + generation loop (`src/sdg/generate.py`)

**Files:**
- Create: `src/sdg/generate.py`
- Test: `tests/test_generate.py`

**Interfaces:**
- Consumes: `sdg.attributes` (`Recipe`, `sample_recipes`); `teacher` (`get_client`, `chat`, `extract_json`); `config` tags.
- Produces:
  - `build_prompt(r: Recipe) -> tuple[str, str]` — returns `(system, user)` strings instructing the teacher to emit strict JSON `{"slang": "...", "english": "..."}` fitting the recipe.
  - `generate_one(client, r: Recipe) -> dict | None` — one teacher call → `{"slang","english"}` dict (attributes merged in), or `None` on parse failure.
  - `main(argv=None) -> int` — CLI: `--limit N` (default `config.SDG_TARGET`), `--pilot` (generate 8, print them, do not write). Writes raw rows to `data/raw/synthetic_slang.raw.jsonl` (validation happens in Task 5).

- [ ] **Step 1: Write the failing test for `build_prompt`**

Create `tests/test_generate.py`:
```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sdg.attributes import Recipe
from sdg.generate import build_prompt


def test_build_prompt_mentions_term_and_json():
    r = Recipe(direction="to_english", term="delulu", tone="playful",
               difficulty="clear", context="texting a friend", is_hard_negative=False)
    system, user = build_prompt(r)
    assert "delulu" in user
    assert "playful" in user
    assert "JSON" in user or "json" in user
    assert '"slang"' in user and '"english"' in user


def test_build_prompt_hard_negative_instruction():
    r = Recipe(direction="to_english", term="fire", tone="deadpan",
               difficulty="edge", context="group chat", is_hard_negative=True)
    _, user = build_prompt(r)
    assert "literal" in user.lower() or "not slang" in user.lower()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_generate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sdg.generate'`.

- [ ] **Step 3: Implement `src/sdg/generate.py`**

Create `src/sdg/generate.py`:
```python
"""Stage 2 — teacher generation. Given a Recipe, ask the NVIDIA teacher for a
natural slang/english pair fitting it. Resumable: appends to a raw jsonl.

Usage:
    uv run python -m sdg.generate --pilot          # 8 examples, printed, not saved
    uv run python -m sdg.generate --limit 1200     # full run, appends to raw jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from config import RAW_DIR, SDG_TARGET
from sdg.attributes import Recipe, sample_recipes
from teacher import chat, extract_json, get_client

RAW_OUT = RAW_DIR / "synthetic_slang.raw.jsonl"

_SYSTEM = ("detailed thinking off\n"
           "You generate training data for a Gen Z/Alpha slang <-> English "
           "translator. Output ONLY a JSON object. No preamble, no markdown.")


def build_prompt(r: Recipe) -> tuple[str, str]:
    if r.is_hard_negative:
        twist = ("Make this a HARD NEGATIVE: write a sentence where the word "
                 f"'{r.term}' is used in its LITERAL, non-slang sense (not slang), "
                 "so the English side is a plain literal reading.")
    else:
        twist = (f"The slang sentence must naturally use the slang term '{r.term}' "
                 f"in its Gen Z sense, with a {r.tone} tone.")
    user = (
        f"Write ONE realistic short message and its translation.\n"
        f"Context: {r.context}. Difficulty: {r.difficulty}.\n"
        f"{twist}\n"
        "Return STRICT JSON with exactly two keys:\n"
        '{"slang": "<the Gen Z slang sentence>", '
        '"english": "<faithful plain-English meaning>"}\n'
        "Both must be single sentences, 3-30 words, no emojis-only."
    )
    return _SYSTEM, user


def generate_one(client, r: Recipe) -> dict | None:
    system, user = build_prompt(r)
    out = chat(client, user, system=system, temperature=0.7, max_tokens=300)
    j = extract_json(out)
    if not j or "slang" not in j or "english" not in j:
        return None
    return {
        "slang": str(j["slang"]).strip(),
        "english": str(j["english"]).strip(),
        "term": r.term, "tone": r.tone, "difficulty": r.difficulty,
        "context": r.context, "is_hard_negative": r.is_hard_negative,
        "direction_focus": r.direction,
    }


def _done_count() -> int:
    if not RAW_OUT.exists():
        return 0
    with open(RAW_OUT, encoding="utf-8") as f:
        return sum(1 for _ in f)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=SDG_TARGET)
    ap.add_argument("--pilot", action="store_true", help="generate 8, print, don't save")
    args = ap.parse_args(argv)

    client = get_client()

    if args.pilot:
        for r in sample_recipes(8, seed=999):
            rec = generate_one(client, r)
            print(json.dumps(rec, ensure_ascii=False, indent=2))
            time.sleep(0.3)
        print("\nPILOT: read these — do the pairs fit the recipe? Then run full.")
        return 0

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    already = _done_count()
    recipes = sample_recipes(args.limit)[already:]  # resume where we left off
    print(f"{args.limit} target | {already} already generated | generating {len(recipes)}")
    with open(RAW_OUT, "a", encoding="utf-8") as f:
        for i, r in enumerate(recipes, 1):
            rec = generate_one(client, r)
            if rec:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            time.sleep(0.25)
            if i % 50 == 0:
                print(f"  {i}/{len(recipes)} ...")
    print(f"\nDone. Raw file: {RAW_OUT}. Next: uv run python -m sdg.validate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run to verify the unit test passes**

Run: `uv run pytest tests/test_generate.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Pilot run (GATED on NVIDIA_API_KEY)**

Run: `uv run python -m sdg.generate --pilot`
(Run from `src/`: `cd src && uv run python -m sdg.generate --pilot`, or add `src` to `PYTHONPATH`.)
Expected: prints 8 JSON pairs. MANUALLY READ them — confirm the slang side uses the term with the right tone, and hard-negatives read literally. If systematically off, fix `build_prompt` before scaling.

- [ ] **Step 6: Commit**

```bash
git add src/sdg/generate.py tests/test_generate.py
git commit -m "feat: Stage 2 teacher generation (prompt builder + resumable loop)"
```

---

### Task 5: Stage 2 — validation + CSV writer (`src/sdg/validate.py`)

**Files:**
- Create: `src/sdg/validate.py`
- Test: `tests/test_validate.py`

**Interfaces:**
- Consumes: `config` (`SDG_PATH`, `EVAL_PATH`, `RAW_DIR`); `prepare_data` (`banned_from_eval`, `load_existing_eval`).
- Produces:
  - `check_row(row: dict, banned: set) -> str | None` — returns a rejection reason string, or `None` if the row is valid. Checks: both fields non-empty; each 3–200 chars; no preamble leak ("sure, here"); featured term present on slang side (skipped for hard negatives); not in `banned` (eval leak).
  - `dedupe_rows(rows: list[dict]) -> list[dict]` — drop case-insensitive duplicate `(slang, english)`.
  - `main(argv=None) -> int` — reads `synthetic_slang.raw.jsonl`, validates, dedupes, writes `data/raw/synthetic_slang.csv` with columns `slang_sentence,normal_sentence,slang_term,tone,difficulty,is_hard_negative`, prints a `kept/rejected(by reason)` summary.

- [ ] **Step 1: Write the failing test**

Create `tests/test_validate.py`:
```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sdg.validate import check_row, dedupe_rows

BANNED = {"she has such beta af energy today.", "extremely beta"}


def ok_row(**kw):
    base = dict(slang="that fit is bussin fr", english="that outfit is excellent",
                term="bussin", is_hard_negative=False)
    base.update(kw)
    return base


def test_valid_row_passes():
    assert check_row(ok_row(), BANNED) is None


def test_empty_field_rejected():
    assert check_row(ok_row(english=""), BANNED) is not None


def test_missing_term_rejected():
    assert check_row(ok_row(slang="totally normal sentence"), BANNED) == "term-missing"


def test_hard_negative_skips_term_check():
    assert check_row(ok_row(slang="the road was closed", term="road",
                            is_hard_negative=True), BANNED) is None


def test_eval_leak_rejected():
    r = ok_row(slang="She has such beta af energy today.")
    assert check_row(r, BANNED) == "eval-leak"


def test_dedupe():
    rows = [ok_row(), ok_row(), ok_row(slang="different one", term="fr")]
    assert len(dedupe_rows(rows)) == 2
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_validate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sdg.validate'`.

- [ ] **Step 3: Implement `src/sdg/validate.py`**

Create `src/sdg/validate.py`:
```python
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_validate.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Full generate + validate (GATED on key; produces the dataset)**

Run (from `src/`): `uv run python -m sdg.generate --limit 1200` then `uv run python -m sdg.validate`
Expected: `synthetic_slang.raw.jsonl` fills, then `data/raw/synthetic_slang.csv` is written with a kept/rejected summary. Skim ~10 rows of the CSV by eye.

- [ ] **Step 6: Commit (code + dataset)**

```bash
git add src/sdg/validate.py tests/test_validate.py data/raw/synthetic_slang.csv
git commit -m "feat: Stage 2 validation + write synthetic_slang.csv"
```

---

### Task 6: Stage 3 — register synthetic source + fold into training

**Files:**
- Modify: `src/config.py` (add one `SOURCES` entry)
- Verify: `data/processed/train.jsonl` grows; `eval.jsonl` unchanged.

**Interfaces:**
- Consumes: the `SOURCES` list schema in `config.py` (`file`, `slang_col`, `english_col`).

- [ ] **Step 1: Register the synthetic source**

In `src/config.py`, add to the `SOURCES` list (after the existing entries):
```python
    {
        "file": "synthetic_slang.csv",
        "slang_col": "slang_sentence",
        "english_col": "normal_sentence",
        # metadata columns (slang_term/tone/difficulty/is_hard_negative) are ignored
        # by load_source; kept in the CSV for provenance.
    },
```

- [ ] **Step 2: Rebuild training data (frozen eval preserved)**

Run: `uv run python src/prepare_data.py`
Expected: summary shows `synthetic_slang.csv -> N usable pairs`, `train.jsonl` example count rises vs. before, and it prints `Using EXISTING frozen eval.jsonl (70 items)` (eval unchanged). Confirm `0` leak warnings.

- [ ] **Step 3: Sanity-check the fold**

Run: `uv run python -c "import json; rows=[json.loads(l) for l in open('data/processed/train.jsonl',encoding='utf-8')]; print('train examples:', len(rows))"`
Expected: a number larger than the pre-synthetic count (previously 42,141).

- [ ] **Step 4: Commit**

```bash
git add src/config.py data/processed/train.jsonl
git commit -m "feat: Stage 3 fold synthetic slang data into training set"
```

- [ ] **Step 5: Retrain (manual / driven) + re-grade — HANDOFF**

This reuses the existing training notebook (no new code). Retrain with:
`RUN_ALL_HEADLESS=1 uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=14400 train_genz_translator.ipynb`
Then: regenerate the grading sheet, and have 2 raters grade for the headline before/after number. Compare per-direction (expect slang→English up, slang direction + abstention not regressed).

---

## Deferred to Phase 2 (separate plan)

**Stage 4 — DPO experiment** is intentionally NOT in this plan. It is gated on the Stage 3 re-grade: we only attempt QLoRA-DPO if the retrained model still shows the "valid-but-rambly phrasing" gap and the classification/translation accuracy is stable. Phase 2 will cover `src/dpo/build_pairs.py` (on-policy rejected → teacher-rewritten chosen → judge → delta-3 gate) and `train_dpo.ipynb` (TRL `DPOTrainer` + QLoRA, `beta=0.1`), with eval for style + accuracy drift.
