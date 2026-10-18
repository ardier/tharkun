# Avengers object-synthesis benchmark

## What the corpus is

The ten pure-Python packages from Lukasczyk et al., *An empirical study of
automated unit test generation for Python* (EMSE 2023) — the Pynguin evaluation
corpus. **Not** the DS-C corpus of Berg et al. (arXiv:2603.06107), which targets
C-extension modules; that paper is followed here for its crash handling and its
statistical protocol, not for its subjects.

`manifest.json` carries two profiles:

| profile | interpreter | versions | comparable to published numbers |
| --- | --- | --- | --- |
| `emse-3.10` | Python 3.10 | the study's own pins | yes |
| `current-3.12` | Python 3.12 | whatever installs today | **no** |

## Running it

```bash
python scripts/setup_benchmarks.py --profile current-3.12 --smoke
python -m tharkun.avengers bench --profile current-3.12 --smoke --repetitions 3
```

One virtual environment per subject: the 2021 pins genuinely conflict with each
other, and a single environment would silently resolve them to something that is
no longer the pinned corpus. Each environment holds the subject, every backend,
Tharkun, and `py_obj_flatting` when a checkout sits beside the repo — so replay
children and backends alike run under the interpreter that can import the
subject.

Results stream to JSON Lines and a campaign resumes, so an interrupted one is
still readable. Repetitions are the outer loop, so a truncated campaign has the
same number of runs per configuration rather than all of one and none of another.

## Smoke result — `current-3.12`

`smoke-current-3.12.txt` is the full report. Its scope, stated up front:
**2 packages, 4 classes, 6 configurations, 3 repetitions, 40 s each — 72 runs.**
That is a smoke run to exercise the harness, not the evaluation. Every number
below is `n = 12` per configuration on `codetiming` and `docstring_parser`, and
none of it is comparable to the published Pynguin numbers.

| configuration | ttfo s | objects | stable states | line | branch | invalid seqs |
| --- | --- | --- | --- | --- | --- | --- |
| builtin | 0.55 | 2 | 1 | 0.449 | 0.385 | 0 |
| hypothesis | 0.53 | 2 | 1 | 0.449 | 0.385 | 0 |
| pynguin | 0.52 | 6.5 | 2 | 0.449 | 0.385 | 1.5 |
| crosshair | 0.51 | 2.5 | 1.5 | 0.449 | 0.385 | 0 |
| ghostwriter | 0.50 | 2 | 1 | 0.449 | 0.385 | 0 |
| **avengers** | **0.49** | **8** | **3** | 0.433 | **0.442** | **2** |

(medians; Mann-Whitney U at α = 0.05 with Vargha-Delaney Â₁₂ in the full report)

**What is significant.** Against the built-in floor, and against `hypothesis`
and `ghostwriter` alone, the portfolio finds more valid objects (8 v 2,
p = 0.046, Â₁₂ = 0.74) and more reproducible receiver states (3 v 1, p = 0.017,
Â₁₂ = 0.77). Both are large effects.

**What is not.** Against `pynguin` and against `crosshair` the portfolio wins
nothing significantly — `pynguin` alone already reaches 6.5 objects. On this
corpus, at this budget, "Avengers beats the best single backend" is **not
established**; what is established is that it beats the ones it can beat while
costing nothing in time to first object (0.49 s v 0.55 s, Â₁₂ = 0.51).

**One number goes the wrong way.** Line coverage is slightly *lower* for the
portfolio (0.433 v 0.449, Â₁₂ = 0.46, not significant). More objects is not the
same as more lines: the extra constructors vary state the class does not branch
on. Branch coverage moves the other way (0.442 v 0.385, also not significant),
which is the more relevant of the two and is what the coverage objective aims
at.

**Why the effects are not sharper.** Twelve runs per configuration is few, two
of the four classes are small, and a 40 s budget saturates the cheap
configurations long before it constrains them — five of six configurations
report `saturated`, so they are not competing for time. The full corpus at the
paper's budget is the measurement; this is the harness working.
