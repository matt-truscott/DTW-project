"""Build an analysis-ready catalog + deterministic user splits for BiosecurID."""

from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
import scipy.io as sio
import numpy as np

from .parse_ids import parse_sig, attempt_label  # expects (user, session, attempt)

# Project root for project-relative paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- internals ---------------------------------------------------------------

def _to_rel(p: Path) -> str:
    """Convert absolute path to project-relative string where possible."""
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(p.resolve())

def _infer_lf_shape(mat_path: Path) -> tuple[int, int]:
    """Return (n_rows, n_cols) for the LocalFunctions matrix in a .mat file."""
    mat = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)

    arr = mat.get("localFunctions", None)
    if arr is None:
        arr = mat.get("LocalFunctions", None)
    if arr is None:
        # case-insensitive fallback
        for k, v in mat.items():
            if isinstance(k, str) and k.lower() == "localfunctions":
                arr = v
                break
    if arr is None:
        raise KeyError(
            f"No localFunctions/LocalFunctions key in {mat_path.name}. "
            f"Keys present: {list(mat.keys())}"
        )

    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if arr.ndim != 2:
        raise ValueError(f"{mat_path.name}: expected 2D LocalFunctions; got shape {arr.shape}")

    return int(arr.shape[0]), int(arr.shape[1])

# --- public API --------------------------------------------------------------

def load_local(mat_path: Path) -> np.ndarray:
    """Load LocalFunctions matrix as a 2D float array (T, 9)."""
    mat = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)

    arr = mat.get("localFunctions", None)
    if arr is None:
        arr = mat.get("LocalFunctions", None)
    if arr is None:
        for k, v in mat.items():
            if isinstance(k, str) and k.lower() == "localfunctions":
                arr = v
                break
    if arr is None:
        raise KeyError(f"No localFunctions/LocalFunctions in {mat_path.name}; keys: {list(mat.keys())}")

    arr = np.asarray(arr)
    arr = np.squeeze(arr)  # drop singleton dims if any

    if arr.ndim != 2:
        raise ValueError(f"{mat_path.name}: expected 2D LocalFunctions; got shape {arr.shape}")

    return arr.astype(float, copy=False)

def build_catalog(
    processed_root: Path,
    output_catalog: Path | None = None,
) -> pd.DataFrame:
    """
    Walk data/processed/uXXXX/{GlobalFeatures,LocalFunctions} and build a single row
    per *attempt* based on LocalFunctions files.

    Output columns:
        path_gf, path_lf, user, session, attempt, label, n_rows_lf, n_cols_lf
    """
    processed_root = Path(processed_root)
    rows: list[dict] = []

    # Iterate per user directory, using LocalFunctions as the driving list
    for udir in sorted(processed_root.glob("u*")):
        lf_dir = udir / "LocalFunctions"
        if not lf_dir.exists():
            continue
        gf_dir = udir / "GlobalFeatures"

        for lf_path in sorted(lf_dir.glob("u*.mat")):
            sid = parse_sig(lf_path.name)            # (user, session, attempt)
            label = attempt_label(sid.attempt)       # 'genuine' or 'skilled'

            gf_path = gf_dir / lf_path.name
            path_gf = _to_rel(gf_path) if gf_path.exists() else ""
            path_lf = _to_rel(lf_path)

            n_rows, n_cols = _infer_lf_shape(lf_path)

            rows.append(
                dict(
                    path_gf=path_gf,
                    path_lf=path_lf,
                    user=sid.user,
                    session=sid.session,
                    attempt=sid.attempt,
                    label=label,
                    n_rows_lf=n_rows,
                    n_cols_lf=n_cols,
                )
            )

    df = pd.DataFrame(rows).sort_values(["user", "session", "attempt"]).reset_index(drop=True)

    if output_catalog is not None:
        output_catalog.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_catalog, index=False, engine="pyarrow")

    return df

def make_splits(df: pd.DataFrame, out_json: Path) -> tuple[list[int], list[int]]:
    """
    Deterministic user split:
      - sort unique user IDs ascending
      - first 300 -> development (train/val)
      - remaining 100 -> evaluation/test
    """
    # Ensure Python-native ints (not numpy.int64) for JSON
    users = sorted(int(u) for u in pd.unique(df["user"].astype(int)))

    dev  = users[:300]
    test = users[300:400]

    payload = {"dev_users": dev, "test_users": test}
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    return dev, test
