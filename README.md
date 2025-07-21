# Dynamic Time Warping (DTW) Project

This repository contains a small collection of utilities for experimenting with the Dynamic Time Warping (DTW) algorithm on signature data stored in MATLAB `.mat` files.

The goal of the project is to provide simple scripts for importing the SVC2004 signature dataset, exploring the data and aligning feature sequences with DTW.

## Environment setup

The recommended way to install the dependencies is via [conda](https://docs.conda.io/):

```bash
conda env create -f environment.yml
conda activate DTW-project
```

Alternatively, you may install the package and its requirements with `pip` after cloning the repository:

```bash
pip install -e .
```

## Basic usage

### Importing `.mat` files

The `src/displayData.py` module shows how to load and inspect MATLAB files using `scipy.io.loadmat`:

```python
from src.displayData import inspect_mat_file

features = inspect_mat_file('data/raw/LocalFunctions/u1001s0001_sg0001.mat')
local_functions = features['localFunctions']
```

If you encounter a MATLAB v7.3 file that `scipy.io` cannot read, use `h5py` instead:

```python
import h5py
with h5py.File('file.mat', 'r') as f:
    data = f['variable'][:]
```

### Running the DTW algorithm

The core DTW implementation lives in `src/dtwAlgorithm.py` as the `dp` function. Given a distance matrix between two sequences it returns the optimal alignment path and the accumulated cost matrix:

```python
import numpy as np
from scipy.spatial.distance import cdist
from src.dtwAlgorithm import dp

# Example toy sequences
a = np.array([1, 2, 3, 4])
b = np.array([1, 1, 2, 3, 5])

# Compute pairwise distances (Euclidean)
dist_mat = cdist(a[:, None], b[:, None])

path, cost = dp(dist_mat)
print('DTW path:', path)
print('Final cost:', cost[-1, -1])
```

### Organising the dataset

`src/importData.py` provides a utility to copy `.mat` files into a structured directory tree. Update `RAW_ROOT` and `PROC_ROOT` in that script to match your local folder layout and then run:

```bash
python src/importData.py
```

## Data Organization

All of the BiosecurID signature features reside under a top-level `data/` folder
(ignored by git for size). It is split into two branches:

```
data/
├── raw/
│   └── LocalFunctions/
│       └── uXXXXsYYYY_sgZZZZ.mat      # variable-length L×9 time-functions
└── processed/
    └── GlobalFeatures/
        └── uXXXXsYYYY_sgZZZZ.mat      # fixed-length 1×40 global feature vector
```

### `raw/LocalFunctions`
Each `.mat` file stores an `nSamples×9` matrix called `localFunctions`
containing the nine selected time-series:

1. x-coordinate
2. y-coordinate
3. pen pressure
4. tangent angle
5. velocity
6. log-curvature
7. acceleration
8. Δx/Δt
9. Δy/Δt

### `processed/GlobalFeatures`
Each `.mat` file stores a 40-dimensional vector named `globalFeatures` that
summarises an entire signature (duration, speed statistics, geometry, pressure
statistics, etc.).

### File naming convention
All files follow the pattern `uXXXXsYYYY_sgZZZZ.mat`, where:

- `XXXX` → user ID (e.g. `1001`)
- `YYYY` → session ID (e.g. `0001`)
- `ZZZZ` → sample number (e.g. `0001`)

Signatures 1, 2, 6 and 7 of each session are genuine, while 3, 4 and 5 are
skilled forgeries.

## Project structure

- `src/` – DTW implementation and data utilities
- `scripts/` – empty placeholders for future helper scripts
- `notebooks/` – exploratory notebooks
- `tests/` – simple examples showing how to load data

## Running tests

This repository contains minimal tests under `tests/`. Run them with:

```bash
pytest
```

(Additional dependencies such as `scipy` may be required for the tests to run.)

