Here’s a refreshed README that matches your current codebase, data layout, and notebook pipeline.

---

# Dynamic Time Warping (DTW) — BiosecurID Signature Verification

This repository contains a complete, reproducible pipeline for **offline signature verification** on **BiosecurID** features using **Dynamic Time Warping (DTW)**. It includes:

* data ingest & integrity checks
* deterministic **dev/test** pairing (genuine / skilled / random)
* fast, parallel DTW with caching + progress bars
* baseline evaluation (AUC, EER, TPR\@FPR)
* **bounded/calibrated** DTW scores
* a **Siamese (tf.keras) DTW** neural extension
* final forgery testing scripts

---
## Visual pipeline (Mermaid)

```mermaid
flowchart LR
  A[LocalFunctions .mat files\n data/processed/uXXXX/LocalFunctions/*.mat] --> B[Pairs (protocol)\n pairs_{dev,test}.parquet]
  B --> C[DTW cache\n dtw_cache_{dev,test}.parquet]
  C --> D[Baseline eval & calibration\n figures/, results/, metrics/]
  B --> E[Load Siamese model\n models/siamese_final.keras]
  E --> F[Siamese inference\n siam_cache_{dev,test}.parquet]
  F --> G[Per-scenario outputs\n metrics/siamese_{scenario}_{split}.csv\n figures/ ROC/DET\n siamese_summary.json]
```

---

## End-to-end I/O table

| Stage | Function(s) | **Inputs** (key columns) | **Outputs** (key columns) | Notes |
|---|---|---|---|---|
| **Data staging** | `src.io.load_biosecurid.load_local` | `.mat` with `localFunctions` (or `LocalFunctions`) | `np.ndarray (T,9)` | Loader used everywhere (DTW+NN). |
| **Catalog & splits** *(optional)* | `build_catalog`, `make_splits` | processed tree | `biosecurid_catalog.parquet` (`path_lf,user,session,attempt,label,…`), `splits.json` | Deterministic dev/test users. |
| **Pairing (protocol)** | `src.pairing.write_pairs_for_splits` | catalog/splits | `pairs_{dev,test}.parquet` → `pair_id, user, query_user, query_session, query_attempt, path_lf_query, path_lf_refs(list[str]), query_label, scenario, split` | Each row = one query + 4 refs. |
| **DTW cache** | `src.dtw.compute_dtw.build_cache` | `pairs_{split}.parquet` | `dtw_cache_{split}.parquet` → `pair_id, d_ref1..4, d_mean, d_min, d_median, len_q,len_r1..r4, backend, window, mode='q2refs'` | Windowed DTW; parallel; cached. |
| **Baseline eval** | `src.evaluation.evaluation.evaluate_scenario` | pairs + DTW cache | ROC/DET figures, metrics JSON/CSV | Uses similarity = `-d_mean`. |
| **Calibration** | `src.calibration.calibration.*` | pairs + DTW cache | `dtw_calibrated_{dev,test}.parquet`, `calibration_summary.json`, figures | Logistic on normalized features. |
| **Siamese training** | `src.nn_utils.data/model/training` | `pairs_meta.parquet` (pairwise) or catalog | `models/siamese_final.keras`, `models/baseline_final.keras` | Sequences resampled to (100×9). |
| **Siamese inference** | `src.nn_utils.infer.score_both_splits` | `pairs_{split}.parquet`, model | `siam_cache_{split}.parquet` → `pair_id, siam_ref1..4, siam_mean, siam_min, siam_median` | Predict q vs 4 refs, aggregate. |
| **Per-scenario outputs** | `src.nn_utils.infer.make_per_scenario_outputs` | pairs + `siam_cache_{split}.parquet` | `metrics/siamese_{scenario}_{split}.csv`, ROC/DET, `siamese_summary.json` | AUC/EER, APCER/BPCER, TPR@FPR.

---

## File locations (by convention)

```
project_root/
├── data/
│   └── processed/
│       ├── pairs_dev.parquet
│       ├── pairs_test.parquet
│       ├── dtw_cache_dev.parquet
│       ├── dtw_cache_test.parquet
│       ├── siam_cache_dev.parquet
│       ├── siam_cache_test.parquet
│       └── metrics/
├── figures/
├── results/
└── models/
    ├── siamese_final.keras
    └── baseline_final.keras
```

---

## Column conventions (most-used)

- **Pairs**: `pair_id`, `path_lf_query` *(str)*, `path_lf_refs` *(list[str])* , `query_label` ∈ {`genuine`,`skilled`,`random`}, `scenario`, `split`.
- **DTW cache**: `pair_id`, `d_ref1..4`, `d_mean`, `d_min`, `d_median`, `len_q`, `len_r1..r4`, `backend`, `window`.
- **Siamese cache**: `pair_id`, `siam_ref1..4`, `siam_mean`, `siam_min`, `siam_median`.

**Scoring polarity**
- Higher = **more genuine**. For DTW, use negative distance (e.g., `score = -d_mean`). Siamese outputs probabilities already on the genuine class.

---

## Minimal end-to-end usage

```python
from pathlib import Path
from src.dtw.compute_dtw import build_cache
from src.nn_utils.infer import score_both_splits, make_per_scenario_outputs

root = Path.cwd()
proc = root/"data"/"processed"

# 1) DTW cache
build_cache(proc/"pairs_dev.parquet",  proc/"dtw_cache_dev.parquet",  window=20)
build_cache(proc/"pairs_test.parquet", proc/"dtw_cache_test.parquet", window=20)

# 2) Siamese inference (requires models/siamese_final.keras)
score_both_splits(proc, model_path=root/"models"/"siamese_final.keras", seq_len=100)

# 3) Scenario metrics/plots
make_per_scenario_outputs(proc, split="dev")
make_per_scenario_outputs(proc, split="test")
```

> **Row-count sanity**: each pairs file has `48 × #users_in_split` rows, and the corresponding DTW/Siam caches must match its `pair_id` set exactly.

## Environment

We recommend Conda:

```bash
conda env create -f environment.yml
conda activate DTW-project
```

Or install in editable mode:

```bash
pip install -e .
```

**Optional accelerators**

* `dtaidistance` → fast multi-dim DTW on CPU
* `dtw_python_cuda` → CUDA DTW (if you have a compatible GPU)

The DTW cache will automatically pick the fastest available backend; you can also force one with `backend="dtaidistance" | "cuda" | "python"`.

---

## Data layout

Place BiosecurID `.mat` feature files under:

```
data/
├── raw/
│   ├── GlobalFeatures/     # 1×40 vector in 'globalFeatures'
│   └── LocalFunctions/     # L×9 matrix in 'localFunctions' (or 'LocalFunctions')
└── processed/              # will be created/populated by the pipeline
```

### File naming invariant

All files follow `uXXXXsYYYY_sgZZZZ.mat`:

* `XXXX` user ID (e.g., `1001`)
* `YYYY` session ID (e.g., `0001`)
* `ZZZZ` attempt index (e.g., `0001`)

**Labeling per attempt index**

* **Genuine**: `{1, 2, 6, 7}`
* **Skilled forgery**: `{3, 4, 5}`

**Sessions per user**: `s0001..s0004`

---

## Project structure (key parts)

```
src/
├── io/
│   ├── load_biosecurid.py     # build catalog/splits, load localFunctions safely
│   └── parse_ids.py           # filename parser + label helpers
├── pairing/
│   └── build_pairs.py         # per-query pairing for dev/test (genuine/skilled/random)
├── dtw/
│   ├── dtwAlgorithm.py        # classic DP kernel for DTW
│   └── compute_dtw.py         # parallel cache builder (progress bars)
├── evaluation/
│   └── evaluation.py          # ROC/AUC/EER + scenario evaluation utilities
├── calibration/
│   └── calibration.py         # normalization + logistic calibrator
├── nn_utils/
│   ├── data.py                # resampling & pair loading for Siamese DTW
│   ├── model.py               # Siamese DTW model (tf.keras) + baseline MLP
│   └── training.py            # training helpers
└── keras_layers/
    └── diff_dtw.py            # differentiable (soft) DTW Keras layer
```

**Notebooks (end-to-end pipeline)**

```
notebooks/
├── 01_ingest.ipynb               # build catalog + splits.json
├── 02_eda.ipynb                  # sanity checks & shapes
├── 03_pairing.ipynb              # pairs_dev/test.parquet
├── 04_dtw_cache.ipynb            # dtw_cache_dev/test.parquet (parallel + progress)
├── 05_eval_baseline.ipynb        # ROC/AUC/EER (legacy pairwise only)
├── 06_bounding_calibration.ipynb # normalize + logistic calibration (q→refs)
├── 07_nn_siamese.ipynb           # Siamese DTW (tf.keras) + baseline NN
└── 08_forgery_testing.ipynb      # run trained model on forged/genuine scenarios
```

---

## Global invariants (used throughout)

* **Filename regex**: `u(\d{4})s(\d{4})_sg(\d{4})` → `(user, session, attempt)`
* **Reference set (per user)**: the **4 genuine** attempts in **session s0001** → refs = `[1,2,6,7]`
* **Genuine queries**: all remaining genuine attempts from sessions `s0002..s0004` (12/user)
* **Skilled queries**: all 12 skilled forgeries (3 per session × 4)
* **Random impostors**: pick **12 other users** (fixed seed), take **one genuine** each (e.g., `s0001_sg0001`)
* **Ref aggregation**: compute DTW to each of 4 refs; use **mean** (keep **min** for ablation)
* **Dev/Test split** (users): sort unique user IDs asc; first **300** → dev, remaining **100** → test → saved to `processed/splits.json`
* **Primary metrics**: ROC AUC, EER, TPR\@FPR∈{0.01, 0.05}, APCER/BPCER (operating point chosen on dev)

---

## Typical outputs

```
data/processed/
├── biosecurid_catalog.parquet
├── splits.json
├── pairs_dev.parquet
├── pairs_test.parquet
├── dtw_cache_dev.parquet
├── dtw_cache_test.parquet
└── metrics/ (optional per-notebook)
figures/
results/
models/
```

---

## Pipeline (notebooks)

1. **01\_ingest** – build catalog & splits

   * Scans `processed/uXXXX/{LocalFunctions,GlobalFeatures}`
   * Validates shapes & labels
   * Writes `biosecurid_catalog.parquet` + `splits.json`

2. **02\_eda** – quick sanity plots

   * LF length distribution, per-user/session label checks
   * Writes `processed/eda_checks.json`

3. **03\_pairing** – generate per-query pairs (dev/test)

   * Writes `pairs_dev.parquet` and `pairs_test.parquet` with:

     ```
     pair_id, scenario, user, query_user, query_session, query_attempt, query_label,
     ref_user, ref_attempts, path_lf_query, path_lf_refs (list[str]), split
     ```

4. **04\_dtw\_cache** – compute & cache DTW distances (parallel)

   * Auto-detects **query→refs** format and caches per-pair:

     ```
     pair_id, query_label, d_ref1..4, d_mean, d_min, d_median,
     len_q, len_r1..r4, backend, window, mode
     ```

   * **Progress** via `tqdm`, **CPU parallelism** via `ProcessPoolExecutor`

   * Example (inside notebook):

     ```python
     from src.dtw.compute_dtw import build_cache
     build_cache(
         pairs_path=PROC_ROOT/"pairs_dev.parquet",
         cache_path=PROC_ROOT/"dtw_cache_dev.parquet",
         backend=None,     # auto: cuda/dtaidistance/python
         window=20,        # Sakoe–Chiba band
         n_jobs=None,      # default: all cores
         overwrite=False,
         show_progress=True
     )
     ```

   * CLI (same detection & behavior):

     ```bash
     python -m src.dtw.compute_dtw \
       data/processed/pairs_dev.parquet \
       data/processed/dtw_cache_dev.parquet \
       --window 20 --backend python
     ```

5. **05\_eval\_baseline** – (legacy pairwise path only)

   * If you still use `pairs_meta.parquet` + `dtw_cache.parquet`, computes ROC/DET.

6. **06\_bounding\_calibration** – normalize + calibrate

   * Loads **query→refs** merges, adds:

     * `d_by_path`, `d_by_ref_len`, `d_by_qry_len`, `d_by_avg_len`
   * Picks best normalization on **dev**, trains **logistic** calibrator, evaluates on **test**
   * Saves figures & JSON metrics

7. **07\_nn\_siamese** – Siamese DTW (tf.keras)

   * Resamples sequences to fixed `T`
   * Trains **Siamese DTW** vs **baseline MLP**
   * Compares with a **classic DTW** reference
   * Saves models to `models/siamese_final.keras`, `models/baseline_final.keras`

8. **08\_forgery\_testing** – run trained model on realistic cases

   * Builds `(ref, target)` scenarios (genuine/skilled/random)
   * Batch predicts with Siamese model; also computes classic DTW distances
   * Outputs per-pair table + quick AUC/EER

---

## Tips: speed & reproducibility

* **Parallel DTW**: `build_cache(..., n_jobs=None)` uses all cores.
* **Progress bars**: enabled by default (`tqdm`).
* **Prevent BLAS oversubscription**: we set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` in `compute_dtw.py` so multi-processing scales predictably.
* **Determinism**: impostor selection & ML seeds are fixed (see code) for reproducible splits & results.

---

## Troubleshooting

* **`ValueError: truth value of an array is ambiguous` when loading `.mat`**
  We use explicit `is None` checks in `load_local(...)`. Keys supported: `localFunctions` or `LocalFunctions` (case-insensitive fallback included).

* **MATLAB v7.3 files**
  If your files are HDF5-based and `scipy.io.loadmat` fails, convert or load via `h5py` and adapt `load_local`.

* **Pylance “module not found” or Keras mix-ups**
  This repo uses **`tf.keras`** consistently. Avoid mixing standalone `keras` with `tf.keras`. If you see a Pylance warning on `DiffDTW`, the code falls back to a safe substitute unless you’re loading a saved model that requires the custom layer (08 notebook enforces import).

---

## Minimal examples

### Parse & load a LocalFunctions matrix

```python
from pathlib import Path
from src.io.load_biosecurid import load_local
arr = load_local(Path("data/processed/u1001/LocalFunctions/u1001s0001_sg0001.mat"))
print(arr.shape)   # (L, 9)
```

### Compute a single DTW distance (classic DP)

```python
import numpy as np
from scipy.spatial.distance import cdist
from src.dtw.dtwAlgorithm import dp

a, b = np.random.randn(120, 9), np.random.randn(105, 9)
dist_mat = cdist(a, b)    # pairwise Euclidean
path, cost = dp(dist_mat)
print(float(cost[-1, -1]))
```

---

## License & dataset

* Code is provided for academic use.
* **BiosecurID** data is not included; ensure you have the right to access and process it.

---

**Enjoy the pipeline!** If you get stuck on any step (pairing rules, DTW caching, Siamese training), open an issue or drop a note in the repo—happy to help tune configs or extend the evaluation.
