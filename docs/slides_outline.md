# Presentation Outline — 5 min, 5–6 slides (every member speaks)

Maps to the assignment's required content (problem → gap → method → results → error analysis →
recommendation) and to your four roles. Aim ~50–55 sec per slide. Practice once and time it.

---

## Slide 1 — Problem (speaker: **Coordinator**)  ~40s
**Title:** Gen Z / Alpha Slang ↔ English Translator
- One model, **both directions**, chosen by a tag:
  - `Translate to English:` → plain English
  - `Translate to Gen Z slang:` → slang
- Input/output example on screen (one each direction).
- Base model: **Llama 3.2 3B Instruct**; method: **QLoRA fine-tuning**.

*Visual:* two arrows (slang→English, English→slang) with a real example each.

---

## Slide 2 — The gap (speaker: **Analysis**)  ~55s
**Title:** Where the base model falls short
- Base model invents/guesses recent slang (2023–2026): `delulu`, `rizz`, `ate`, `villain era`…
- **Show ONE concrete base-model failure** (copy from section 4 of the notebook):
  > IN: "bro really ate with that fit, delulu fr"
  > BASE (wrong): "…" ← its confidently-wrong answer
  > CORRECT: "he genuinely looked great in that outfit, he's kidding himself for real"
- Point: this is a **knowledge + behaviour gap** → fine-tuning is the right tool (not RAG).

*Visual:* the failing example, base answer in red, correct answer in green.

---

## Slide 3 — What we did (speaker: **Modelling**)  ~55s
**Title:** SFT / QLoRA on Llama 3.2 3B
- **Data:** ~15.8k clean slang↔English pairs from Kaggle/HF sources, deduped, both directions
  (→ 31.5k training examples). 60-item eval set frozen out (never trained on).
- **Technique:** LoRA adapters (r=16) on the 4-bit base — fits a T4 / trains fast on our GPU.
- **Why SFT not DPO/RAG:** the gap is *right vs wrong meaning* (knowledge), not *style preference*,
  and we want the skill in the weights.
- Keep it high-level — one line on each.

*Visual:* tiny pipeline diagram: raw data → prep → QLoRA → adapter.

---

## Slide 4 — Results (speaker: **Data & eval**)  ~55s
**Title:** Before vs after (honest numbers)
- **Primary metric = human accuracy** (2 raters), correct/incorrect vs reference meaning.
- Table: accuracy **base vs tuned**, per direction + overall (fill from the scoring cell).

  | | Base | Tuned |
  |---|---|---|
  | Slang → English | __% | __% |
  | English → Slang | __% | __% |
  | Overall | __% | __% |

- Inter-rater agreement: __%.
- **Be honest:** name where it *didn't* improve (e.g. one direction lagged, or rare terms).
- *(Optional)* mention the local 8B judge agreement if you ran it.

*Visual:* a simple base-vs-tuned bar chart.

---

## Slide 5 — Error analysis (speaker: **Analysis**)  ~45s
**Title:** What it still gets wrong
- 1–2 tuned-model failure cases (from the grading sheet) + *why*:
  - e.g. very new/rare slang not in training data.
  - e.g. over-literal English→slang (inserts the wrong slang term).
- Shows you understand the model, not just the score.

*Visual:* 1–2 failing examples with a one-line reason each.

---

## Slide 6 — Recommendation (speaker: **Coordinator**)  ~40s
**Title:** Deploy / Iterate / Hold — our call
- State the call clearly (likely **Iterate**: it improved meaningfully but rare-slang coverage
  and one direction need more data) and **defend it** with the numbers from slide 4.
- One line on next step (e.g. add real scraped pairs from `final dataset.xlsx`).
- *(Optional)* live 30-sec Gradio demo if time allows — **only if practiced and safe**.

*Visual:* big DEPLOY / ITERATE / HOLD with your choice circled + the one reason.

---

### Speaking split (everyone speaks — a grading requirement)
- **Coordinator:** slides 1 + 6
- **Analysis:** slides 2 + 5
- **Modelling:** slide 3
- **Data & eval:** slide 4

### Timing / delivery reminders
- 5 minutes total, then 2 min Q&A. **Going over time counts against you** — practice once with a timer.
- Max 6 slides. Keep text tight; talk to the visuals.
- Have the notebook open in a tab in case Q&A asks to see a result.
