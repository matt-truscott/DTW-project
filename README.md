# Dynamic Time Warping (DTW) — BiosecurID Signature Verification

A complete, reproducible pipeline for **offline signature verification** on **BiosecurID** using **Dynamic Time Warping (DTW)** with a neural **Siamese DTW** extension.

**What you get**
- Data ingest & integrity checks
- Deterministic **dev/test** pairing (genuine / skilled / random)
- Fast, parallel DTW with caching + progress bars
- Baseline evaluation (AUC, EER, TPR@FPR)
- Bounded/normalized + **logistic calibration**
- **Siamese (tf.keras) DTW** model + inference
- Forgery scenario testing & summary metrics

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

| Stage                             | Function(s)                                    | **Inputs** (key columns)                           | **Outputs** (key columns)                                                                                                                                    | Notes                              |
| --------------------------------- | ---------------------------------------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------- |
| **Data staging**                  | `src.io.load_biosecurid.load_local`            | `.mat` with `localFunctions` (or `LocalFunctions`) | `np.ndarray (T,9)`                                                                                                                                           | Loader used everywhere (DTW + NN). |
| **Catalog & splits** *(optional)* | `build_catalog`, `make_splits`                 | processed tree                                     | `biosecurid_catalog.parquet` (`path_lf,user,session,attempt,label,…`), `splits.json`                                                                         | Deterministic dev/test users.      |
| **Pairing (protocol)**            | `src.pairing.write_pairs_for_splits`           | catalog/splits                                     | `pairs_{dev,test}.parquet` → `pair_id, user, query_user, query_session, query_attempt, path_lf_query, path_lf_refs(list[str]), query_label, scenario, split` | One row = one query + 4 refs.      |
| **DTW cache**                     | `src.dtw.compute_dtw.build_cache`              | `pairs_{split}.parquet`                            | `dtw_cache_{split}.parquet` → `pair_id, d_ref1..4, d_mean, d_min, d_median, len_q,len_r1..r4, backend, window, mode='q2refs'`                                | Windowed DTW; parallel; cached.    |
| **Baseline eval**                 | `src.evaluation.evaluation.evaluate_scenario`  | pairs + DTW cache                                  | ROC/DET figures, metrics JSON/CSV                                                                                                                            | Similarity = `-d_mean`.            |
| **Calibration**                   | `src.calibration.calibration.*`                | pairs + DTW cache                                  | `dtw_calibrated_{dev,test}.parquet`, `calibration_summary.json`, figures                                                                                     | Logistic on normalized features.   |
| **Siamese training**              | `src.nn_utils.data/model/training`             | `pairs_meta.parquet` (pairwise) or catalog         | `models/siamese_final.keras`, `models/baseline_final.keras`                                                                                                  | Sequences resampled to (100×9).    |
| **Siamese inference**             | `src.nn_utils.infer.score_both_splits`         | `pairs_{split}.parquet`, model                     | `siam_cache_{split}.parquet` → `pair_id, siam_ref1..4, siam_mean, siam_min, siam_median`                                                                     | Predict q vs 4 refs, aggregate.    |
| **Per-scenario outputs**          | `src.nn_utils.infer.make_per_scenario_outputs` | pairs + `siam_cache_{split}.parquet`               | `metrics/siamese_{scenario}_{split}.csv`, ROC/DET, `siamese_summary.json`                                                                                    | AUC/EER, APCER/BPCER, TPR@FPR.     |

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

## Column conventions

* **Pairs**: `pair_id`, `path_lf_query` *(str)*, `path_lf_refs` *(list[str])* , `query_label` ∈ {`genuine`,`skilled`,`random`}, `scenario`, `split`.
* **DTW cache**: `pair_id`, `d_ref1..4`, `d_mean`, `d_min`, `d_median`, `len_q`, `len_r1..r4`, `backend`, `window`.
* **Siamese cache**: `pair_id`, `siam_ref1..4`, `siam_mean`, `siam_min`, `siam_median`.

**Scoring polarity**
Higher = **more genuine**. For DTW, use a negative distance (e.g., `score = -d_mean`). Siamese outputs probabilities on the genuine class.

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

> **Row-count sanity**: each pairs file has `48 × #users_in_split` rows. The DTW/Siam caches must match its `pair_id` set exactly.

---

## Environment

Use Conda:

```bash
conda env create -f environment.yml
conda activate DTW-project
```

Or editable install:

```bash
pip install -e .
```

**Optional accelerators**

* `dtaidistance` → fast multi-dim DTW on CPU
* `dtw_python_cuda` → CUDA DTW (if supported)

DTW cache picks the fastest available backend automatically. You can force one: `backend="dtaidistance" | "cuda" | "python"`.

---

## Data layout

Place BiosecurID `.mat` feature files under:

```
data/
├── raw/
│   ├── GlobalFeatures/     # 1×40 vector in 'globalFeatures'
│   └── LocalFunctions/     # L×9 matrix in 'localFunctions' (or 'LocalFunctions')
└── processed/              # created/populated by the pipeline
```

**Filename pattern**: `uXXXXsYYYY_sgZZZZ.mat`

* `XXXX` user (e.g., `1001`), `YYYY` session (`0001..0004`), `ZZZZ` attempt
* **Genuine**: `{1,2,6,7}`; **Skilled forgery**: `{3,4,5}`

---

## Project structure (key parts)

```
src/
├── io/
│   ├── load_biosecurid.py     # catalog/splits, load LocalFunctions safely
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

**Notebooks**

```
notebooks/
├── 01_ingest.ipynb               # build catalog + splits.json
├── 02_eda.ipynb                  # sanity checks & shapes
├── 03_pairing.ipynb              # pairs_{dev,test}.parquet
├── 04_dtw_cache.ipynb            # dtw_cache_{dev,test}.parquet
├── 05_eval_baseline.ipynb        # ROC/AUC/EER (legacy pairwise only)
├── 06_bounding_calibration.ipynb # normalize + logistic calibration (q→refs)
├── 07_nn_siamese.ipynb           # Siamese DTW (tf.keras) + baseline NN
└── 08_forgery_testing.ipynb      # run trained model on realistic scenarios
```

---

## Global invariants

* **Reference set (per user)**: 4 genuine attempts in session **s0001** → `[1,2,6,7]`
* **Genuine queries**: remaining genuine attempts from sessions `s0002..s0004` (12/user)
* **Skilled queries**: 12 skilled forgeries (3 per session × 4)
* **Random impostors**: pick **12 other users** (fixed seed), take one genuine each
* **Aggregation**: compute against the 4 refs; use **mean** (also keep min/median)
* **Dev/Test split** (users): sort IDs; first **300** → dev, remaining **100** → test (`processed/splits.json`)
* **Primary metrics**: ROC AUC, EER, TPR@FPR∈{0.01, 0.05}, APCER/BPCER (operating point from dev)

---

## Tips: speed & reproducibility

* **Parallel DTW**: `build_cache(..., n_jobs=None)` uses all cores.
* **Progress bars**: via `tqdm`.
* **Avoid BLAS oversubscription**: `compute_dtw.py` sets `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`.
* **Determinism**: fixed RNG for impostor selection and ML seeds.

---

## Troubleshooting

* **`ValueError: truth value of an array is ambiguous` when loading `.mat`**
  We use explicit `is None` checks in `load_local(...)`. Keys supported: `localFunctions` or `LocalFunctions` (case-insensitive fallback included).

* **MATLAB v7.3 files**
  If `scipy.io.loadmat` fails (HDF5-based), load via `h5py` and adapt `load_local`.

* **Pylance “module not found” or Keras mix-ups**
  This repo uses **`tf.keras`** consistently. Avoid mixing standalone `keras`. If you load a saved model that requires the custom `DiffDTW` layer, ensure `src/keras_layers/diff_dtw.py` is importable.

---

## Minimal examples

**Parse & load a LocalFunctions matrix**

```python
from pathlib import Path
from src.io.load_biosecurid import load_local
arr = load_local(Path("data/processed/u1001/LocalFunctions/u1001s0001_sg0001.mat"))
print(arr.shape)   # (L, 9)
```

**Compute a single DTW distance (classic DP)**

```python
import numpy as np
from scipy.spatial.distance import cdist
from src.dtw.dtwAlgorithm import dp

a, b = np.random.randn(120, 9), np.random.randn(105, 9)
dist_mat = cdist(a, b)          # pairwise Euclidean
path, cost = dp(dist_mat)
print(float(cost[-1, -1]))
```

---

## License & dataset

* Code is provided for academic use.
* **BiosecurID** data is not included; ensure you have the right to access and process it.
