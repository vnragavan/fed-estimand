# fed-estimand
Federated learning experiments on FLamby's Fed-Heart-Disease dataset that ask
what *estimand* the federation is computing — patient-weighted vs.
site-weighted, pooled discrimination vs. per-site calibration, and how that
estimand deforms under differential privacy.

The repository contains a reproducible CPU-only experiment kit built on
[Flower](https://github.com/adap/flower) and
[FLamby](https://github.com/owkin/FLamby), three end-to-end experiments
(Pareto frontier, per-site calibration, and DP privacy–utility tradeoff).
**Only source is versioned** — models, predictions, logs, tables, and figures
under `results/` and `figures/` are gitignored; run **`make all`** after setup
to regenerate them.

---

## Repository layout

```
fed-estimand-workshop/
├── Makefile                      # make all | clean | fresh (see README)
├── scripts/clean_results.sh      # remove generated results/ + figures/ (see make clean)
├── src/                          # data, model, client, train, strategies, metrics, calibration, flamby_sites, ...
├── experiments/
│   ├── exp1_sweep.py             # FedAvg α sweep (5 seeds × 3 α); writes results/exp1_results.*
│   ├── exp1_figure.py            # bootstrap CI summary + figures/exp1.{pdf,png}
│   ├── exp3_calibration.py       # per-site calibration (FedAvg α=1 checkpoint from Exp 1)
│   ├── exp5_sweep.py             # DP training sweep (4 ε × 3 seeds)
│   └── exp5_figure.py            # paper-style DP figure + LaTeX snippets
├── figures/                      # populated by experiments (.gitignore: *.pdf, *.png)
├── results/                      # populated by experiments (models, JSON, logs — gitignored)
├── requirements.txt
├── setup_env.sh
└── README.md
```

---

## Setup

This kit targets **Apple Silicon CPU** (no MPS, no CUDA). The CPU constraint is
hard-coded in `src/data.py`, `src/model.py`, `src/client.py`, `src/train.py`,
and `src/eval_per_site.py` via `torch.set_default_device("cpu")`. It runs on
any x86 Linux/macOS box too — only the wall-clock estimates change.

### 1. Create the virtual environment

```bash
bash setup_env.sh
source .venv/bin/activate
```

`setup_env.sh` creates `.venv/`, installs everything in `requirements.txt`
(Flower, PyTorch, Opacus, FLamby, NumPy<2, scikit-learn, matplotlib, tqdm),
and prints the installed versions of every dependency.

### 2. Download Fed-Heart-Disease

FLamby's heart-disease loader requires a one-time data download. Follow the
instructions in
[`flamby/datasets/fed_heart_disease`](https://github.com/owkin/FLamby/tree/main/flamby/datasets/fed_heart_disease).
On macOS you may need to point `SSL_CERT_FILE` at your `certifi` bundle:

```bash
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
python -m flamby.datasets.fed_heart_disease.dataset_creation_scripts.download \
    --output-folder ./fed_heart_data
```

### 3. Verify

```bash
python -c "from src.data import get_site_sizes; print(get_site_sizes())"
# expected: {0: 199, 1: 172, 2: 30, 3: 85}
```

If this prints the four site sizes, you're ready to run experiments.

### One-shot reproduction (all three experiments)

From `fed-estimand-workshop/` after `setup_env.sh` and the data download:

```bash
make all
```

This runs Experiment 1, then 3, then 5 in order and checks that the main PDF/JSON artifacts exist. Use `make help` for individual targets.

To **delete all generated results and figures** (models, predictions, logs, CSV/JSON, PDFs/PNGs) and re-run from scratch:

```bash
make fresh
```

(`make clean` only removes artifacts; inspect with `bash scripts/clean_results.sh --dry-run`.)

---

## Running the experiments

All commands assume `source .venv/bin/activate` has been run.

### Experiment 1 — patient-vs-site Pareto frontier

Sweeps the FedAvg weighting exponent `α ∈ {0.0, 0.5, 1.0}` over 5 seeds (15
runs total). `α=0` weights every site equally; `α=1` is FedAvg's default
sample-size weighting.

```bash
python -m experiments.exp1_sweep
python -m experiments.exp1_figure               # CI summary + plot
```

Outputs:
- `results/exp1_results.{csv,json}`, `results/exp1_ci_summary.csv`
- `figures/exp1.{png,pdf}` — patient-weighted vs. worst-site loss,
  with seed-SE error bars and a footnote reporting average bootstrap-over-patients
  CI half-widths.

If an older checkout still has `exp1_v2_results.*` or `exp1_v2_ci_summary.csv`,
rename them to `exp1_results.*` and `exp1_ci_summary.csv`, or rerun `make exp1`.

### Experiment 3 — pooled vs. per-site calibration

Loads the FedAvg-default (`α=1`) model from Experiment 1 and computes
per-site calibration curves with Wilson 95 % CIs and bootstrap CIs (with
`C=1.0` regularised logistic recalibration to avoid perfect-separation
artifacts on small sites).

```bash
python -m experiments.exp3_calibration
# Re-render figures from results/exp3_results.json only (no model / no bootstrap):
python -m experiments.exp3_calibration --refigure-only
```

Outputs:
- `results/exp3_summary.csv` and `results/exp3_results.json`
- `figures/exp3.{png,pdf}` — 2 × 2 grid, one panel per site
- Across-α robustness table: per-site mean ± SE of `calib_intercept` and
  `calib_slope` at α ∈ {0, 0.5, 1.0} across 5 seeds.

### Experiment 5 — DP privacy–utility tradeoff

Re-trains the model under server-side fixed-clipping DP at `ε ∈ {1, 3, 10, ∞}`
with `δ=1e-5` and 3 seeds per budget (12 runs). Noise multipliers come from
Opacus's RDP accountant.

```bash
python -m experiments.exp5_sweep               # full sweep, ~2.5 min
python -m experiments.exp5_figure               # re-render paper figure + LaTeX, ~2 s
```

Outputs:
- `results/exp5_results.{csv,json}`, `results/diagnostics_dp.log`
- `figures/exp5_original.{png,pdf}` — written at the end of the sweep as a fast sanity check (pooled AUC + worst-|intercept|); you get a plot as soon as training finishes without running another script.
- `figures/exp5.{png,pdf}` — workshop/paper figure from `exp5_figure.py`: pooled vs. worst-site AUC plus worst-site calibration slope deviation (different layout and metrics than the diagnostic).
- `results/exp5_paper_{text,figure}.tex` — LaTeX-ready snippets

---

## Reproducibility

The kit takes reproducibility seriously because the central claims rest on
small absolute differences across (α, seed) pairs.

- **Independent per-client seeds.** `run_federation` uses
  `numpy.random.SeedSequence(seed).spawn(NUM_CLIENTS)` to derive a genuinely
  independent child seed for each client. This avoids the `seed + cid`
  collision pattern where seed `s` and seed `s+1` end up producing the same
  per-client init for clients shifted by one.
- **Three RNGs are seeded** (`random`, `numpy`, `torch`) at every level
  (top-of-run and per-client), and the deterministic-cuDNN flags are set even
  though the kit runs on CPU.
- **Diagnostics logs.** Every run appends a line to:
  - `results/diagnostics_seeds.log` — the `torch.initial_seed()` after seeding
  - `results/diagnostics_init.log` — a hash of each client's initial parameters
    on round 1 (different seed → different hashes; same seed but different α →
    same hashes, since α changes only the aggregation weight)
  - `results/diagnostics_alpha_weights.log` — the per-round normalized
    aggregation weights (`p_k = n_k^α / Σ`)
  - `results/diagnostics_dp.log` — `(ε, δ, sample_rate, epochs, noise_multiplier,
    clip_norm)` for every DP run
- **Bootstrap and seed-SE are reported separately.** Confidence intervals over
  patients (`src/bootstrap_ci.py`, `src/calibration.py`) capture
  population-loss uncertainty; standard errors across seeds capture
  training stochasticity. The figures show whichever is load-bearing for the
  claim being made and report the other in the caption.

To verify a fresh checkout reproduces published numbers, run the experiment
scripts above; the `seed=0` row in `exp1_results.csv` should be bit-for-bit
identical across independent runs of the same code (modulo logging
timestamps).

---

## Hardware target

CPU only, on Apple Silicon (M-series) or x86. PyTorch's MPS and CUDA backends
are intentionally not used; mixing them with Opacus and Flower's DP wrappers
is fragile, and on a four-site federation with a tiny MLP the GPU path
doesn't pay off anyway.

---

## Notes on the implementations

- **`AlphaWeightedFedAvg`** subclasses `flwr.server.strategy.FedAvg` and
  overrides `aggregate_fit` to weight client updates by `n_k^α / Σ`. It
  records the realized weights to `diagnostics_alpha_weights.log` for every
  round, so you can verify post-hoc that α actually changed the aggregation.
- **DP wrapper.** `make_dp_strategy` wraps a base strategy with
  `DifferentialPrivacyServerSideFixedClipping` and a captured-parameters
  subclass so that `run_federation` can persist the *post-noise* aggregate
  (Flower's vanilla wrappers expose only the pre-noise inner-strategy
  parameters). The Opacus accountant computes the noise multiplier from
  `(ε, δ, sample_rate, num_rounds)`.
- **Numerical guard for DP.** `src/strategies.py` patches Flower's
  `clip_inputs_inplace` to no-op when the model update has zero L2 norm.
  Heavy noise (e.g., `ε=1`) can saturate the sigmoid output, vanish gradients,
  and produce identical client returns; the unpatched function divides by the
  zero norm and crashes. The guard is mathematically equivalent to
  `scaling_factor = 1` (a zero vector scaled by anything is still zero).
- **Calibration on small sites.** The Switzerland center (FLamby site 2, 16 test patients) is
  small enough that unregularised logistic recalibration produces extreme
  slopes/intercepts on bootstrap resamples that admit perfect separation.
  Both `compute_calibration_intercept_slope` (point estimate) and
  `bootstrap_calibration_intercept_slope` use `C=1.0`; the bootstrap also
  drops resamples whose `|intercept|` or `|slope|` exceeds 10.

---

## License

MIT — see [LICENSE](LICENSE). Fed-Heart-Disease itself is not shipped in this
repo; it sits behind FLamby's [redistribution terms](https://github.com/owkin/FLamby/blob/main/LICENSE)
and is fetched locally by `setup_env.sh`.

## Built on

- [Flower](https://github.com/adap/flower) — federation simulation and DP wrappers
- [FLamby](https://github.com/owkin/FLamby) — Fed-Heart-Disease loader and the per-site train/test splits
- [Opacus](https://github.com/pytorch/opacus) — the RDP accountant used to derive DP noise multipliers
