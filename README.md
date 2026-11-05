# Phishing URL Detection from Lexical and Host-Based Features

A self-contained research mini-project: machine-learning classification of
URLs as **phishing** or **legitimate** using only features computed from the
URL string (no DNS, WHOIS, page fetch or reputation lookups). It contains the
code, a reproducible pipeline, all results and figures, and a written paper.

| | |
|---|---|
| Paper | [`paper/paper.md`](paper/paper.md) |
| One-command reproduction | `python run_pipeline.py --external` |
| Results (tables + figures) | [`results/`](results/) |
| Dataset (generated) | [`data/synthetic_urls.csv`](data/synthetic_urls.csv) |

## Headline results

On a held-out test set of 4,001 URLs whose **registered domains never appear
in training**:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.939 | 0.950 | 0.926 | 0.938 | 0.985 | 4.8% |
| Random Forest | 0.966 | 0.962 | 0.970 | 0.966 | 0.996 | 3.8% |
| **Gradient Boosting** | **0.968** | 0.962 | **0.975** | **0.968** | **0.997** | 3.9% |

These in-distribution scores are high, but the extra experiments show they
do not carry over to realistic conditions:

* **Real phishing URLs:** on 5,000 URLs from a public feed, recall falls to
  **39–41%**. Only 37% of real phishing URLs show any "textbook" indicator
  (IP host, `@`, suspicious TLD, brand name, look-alike, punycode, login-type
  word), compared with 90% of the synthetic ones.
* **Adaptive attacker:** if the attacker avoids brand names and login-type
  words, detection falls from 95% to **0–4%**.
* **Base rate:** at a realistic 1% phishing prevalence, a 3.9% false-positive
  rate means only **~20% of alerts are real phishing**.

See the paper for the full analysis.

## Setup

Requires Python ≥ 3.10 (tested on 3.13).

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

## Running

Reproduce everything (synthetic data, models, metrics, figures, and the
real-URL check, which downloads ~66 MB once):

```bash
python run_pipeline.py --external
```

Fully offline (everything except the real-URL check):

```bash
python run_pipeline.py
```

Unit tests:

```bash
python -m unittest discover -s tests -v
```

A full run takes about 1.5 minutes on a laptop-class CPU. For a fixed
`--seed` (default 42) every output file is identical across runs; this was
verified by hashing `results/` and `data/` over three consecutive runs.

The committed results were produced with `--external`. An offline run
regenerates every file except the `external_*.csv` tables, and the
`metrics.json` it writes has no `external_check` block.

Options: `--seed`, `--n-per-class` (default 10,000), `--n-external`
(default 5,000), `--out`.

## Project layout

```
run_pipeline.py            one entry point that regenerates data/ and results/
src/phishdet/
  constants.py             brand → official domains, suspicious TLDs, shorteners,
                           public/private suffixes, sensitive words
  features.py              URL parsing + 25 lexical / host-based features
  data_gen.py              deterministic synthetic URL generator (both classes)
  models.py                logistic regression, random forest, gradient boosting
  metrics.py               metrics incl. base-rate-adjusted precision
  evasion.py               adaptive-attacker test sets (levels L0–L4)
  external.py              optional check on real phishing URLs (pinned feed)
  plots.py                 all figures
  pipeline.py              orchestrates the experiment
tests/test_features.py     unit tests (parsing, features, generator constraints)
data/synthetic_urls.csv    generated corpus with technique labels and split
results/                   CSV/JSON tables and figures/ (see below)
paper/paper.md             the research paper
```

### Outputs in `results/`

| File | Contents |
|---|---|
| `metrics_comparison.csv` | Test metrics per model (accuracy, precision, recall, F1, ROC-AUC, AP, FPR, precision at 1% prevalence, confusion counts) plus 5-fold CV mean ± std |
| `confusion_matrices.csv` | Confusion matrices in long format |
| `cv_results.csv` | Grouped 5-fold cross-validation on the training set |
| `metrics_random_split.csv` | Same models with a naive random split, for comparison |
| `error_by_category.csv` | Detection rate per phishing technique / FPR per legitimate category |
| `feature_importance.csv`, `logreg_coefficients.csv` | Permutation importance (all models), LR coefficients |
| `ablation.csv` | Lexical-only vs host-only vs all features |
| `evasion.csv` | Detection rate under adaptive-attacker levels L0–L4 |
| `external_check*.csv`, `external_feature_shift.csv` | Real-phishing recall, indicator breakdown, feature means synthetic vs real |
| `dataset_summary.csv`, `metrics.json` | Corpus composition; machine-readable summary |
| `figures/*.png` | ROC, PR, confusion matrices, feature importance, feature distributions, error by category, ablation, evasion |

## Dataset: why synthetic, and how it was made

**Tried first:** the UCI PhiUSIIL phishing URL dataset, the PhishTank feed
and the Tranco top-1M list. From the build environment the outbound proxy
refused all three (HTTP 403). The only reachable source was
[Phishing.Database](https://github.com/mitchellkrogza/Phishing.Database),
which lists **phishing URLs only**.

Mixing those real phishing URLs with legitimate URLs from some other source
would create a **source-confound**: the model could learn "which list did this
come from" (crawl-era formatting, path styles, TLD mix) instead of "is this
phishing". So the main experiment uses a **synthetic corpus in which both
classes come from one generator** (`src/phishdet/data_gen.py`), and the real
feed is used only as an **external recall check**.

The generator (seeded, deterministic) produces 10,000 URLs per class:

* **Phishing** URLs use 11 documented techniques: IP-address hosts, `@`
  userinfo tricks, long subdomain chains (`paypal.com.verify-login.x.tk`),
  typosquats/homoglyphs (`paypa1`, `rnicrosoft`), brand + keyword combos
  (`secure-amazon-update.info`), random domains on cheap TLDs, free-hosting
  platforms (`*.web.app`, `*.pages.dev`), compromised small-business sites
  (`/wp-content/plugins/.../login.php`), URL shorteners, punycode, and
  clean "professional" newly-registered domains.
* **Legitimate** URLs come from 8 categories: well-known sites, long-tail
  small sites, institutions, CDN/cloud storage, projects on free-hosting
  platforms, shortener links, intranet/router IPs, and articles that
  *mention* a brand.
* **Hard cases are deliberate:** legitimate HTTP sites, hyphenated or
  digit-containing names, cheap TLDs, random-looking CDN hosts, real login
  pages and `/@user` paths; phishing on HTTPS, compromised benign domains and
  brand-free domains.

Every row carries its generating technique, so errors can be broken down by
technique (`results/error_by_category.csv`).

**Main caveat:** a synthetic corpus encodes the designer's beliefs about
phishing, so in-distribution scores are optimistic. The external check
measures how optimistic: about 97% recall in-distribution vs about 40% on
real phishing URLs. The generator was frozen before the external check was
first run and was not tuned to it.

The raw real-URL feed (live malicious links) is cached in `data/external/`,
which is **git-ignored**; only aggregate statistics are committed.

## Design choices

* **String-only features (25 in total: 12 lexical, 13 host-based).** They
  need no network access, so they are fast, private (nothing leaks to
  third-party lookups) and usable offline in mail gateways or browsers. They
  include all the requested features: length, dots/hyphens/digits, IP host,
  subdomain count, `@`, HTTPS, suspicious TLD, Shannon entropy of the
  hostname, brand presence, and path depth. They also include
  domain-identity features that sharpen the brand signal:
  *brand outside its official domain* (`paypal` in a URL not on
  `paypal.com`) and *brand look-alike* (homoglyph normalisation + edit
  distance).
* **No third-party URL libraries.** Registered domains are computed with a
  small built-in subset of the [Public Suffix List](https://publicsuffix.org)
  (including private suffixes such as `github.io` and `web.app`, so each
  customer subdomain is its own "site", as browsers treat it). Dependencies
  stay limited to numpy, pandas, scikit-learn and matplotlib.
* **Domain-disjoint split.** The 80/20 test split is grouped by registered
  domain (`StratifiedGroupKFold`), so no domain appears in both train and
  test. This measures generalisation to *new* domains, which is what a
  detector meets in a fresh phishing campaign. A naive random split is
  reported alongside for comparison.
* **Three models of increasing capacity.** Logistic regression (log1p +
  standardisation; interpretable baseline), random forest, and histogram
  gradient boosting. Hyper-parameters are fixed, sensible defaults chosen up
  front, never tuned on the test set; grouped 5-fold CV on the training set
  reports their stability.
* **Operational metrics.** Besides accuracy/F1/ROC-AUC we report FPR and
  **precision at 1% prevalence**, because a balanced test set hides the
  base-rate problem of real deployments.
* **Model-agnostic importance.** Permutation importance on the test set
  (drop in ROC-AUC) is used for every model, plus LR coefficients.

## Threat model and assumptions

* **Setting:** a filter that scores URLs (e.g. links in e-mail or chat)
  *before* they are visited, seeing only the URL string.
* **Attacker capabilities assumed in the main experiment:** the attacker uses
  common, off-the-shelf tricks and does not know the detector's features.
  This is the "non-adaptive" setting most published results implicitly
  assume.
* **Adaptive attacker (evasion experiment):** the attacker knows the feature
  set and can buy a `.com`, get a free TLS certificate, avoid IPs/`@`/
  shorteners, leave brand names and login words out of the URL, and finally
  host the kit on an ordinary-looking (e.g. compromised) site. All of these
  are cheap. Results show the detector is **not robust** to this attacker.
* **Out of scope:** page content, visual similarity, redirect chains, domain
  age/WHOIS, DNS, TLS certificate details, and hosting reputation. These
  signals are harder for an attacker to fake and are the natural next layer.
* **Intended use:** a cheap first-stage triage score inside a layered defence,
  not a standalone verdict.

## License and data ethics

Code is provided for research and education. The synthetic corpus contains
fabricated URLs; some typosquat-style names may coincide with real
registered domains, so do not visit them. Real phishing URLs are never
committed to this repository.
