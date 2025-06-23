from pathlib import Path
import numpy as np
import scipy.io as sio   # SciPy loader
import h5py              # only used if you hit v7.3 files

RAW_ROOT = Path(r"C:\Users\mattt\Skripsie\Projects\DTW-project\data\raw")
G_DIR, L_DIR = RAW_ROOT / "GlobalFeatures", RAW_ROOT / "LocalFunctions"

def list_ids():
    """All signature IDs without .mat suffix, sorted."""
    return sorted(p.stem for p in G_DIR.glob("*.mat"))

# ---------- helpers ---------------------------------------------------------
def _load_mat(path: Path, key: str | None = None):
    """Return np.ndarray stored under `key` (first top-level array if None)."""
    try:                                           # v5 / v7
        mat = sio.loadmat(path, squeeze_me=True)
        if key is None:            # choose first non-meta variable
            key = next(k for k in mat if not k.startswith("__"))
        return np.asarray(mat[key])
    except NotImplementedError:                    # v7.3  (HDF5-based)
        with h5py.File(path, "r") as f:
            if key is None:
                key = next(iter(f.keys()))
            return np.asarray(f[key])

# ---------- public API ------------------------------------------------------
def load_global(sig_id: str, key: str | None = None) -> np.ndarray:
    """
    Return shape (49,) global-feature vector.
    If you don't know the variable name, leave `key=None`.
    """
    return _load_mat(G_DIR / f"{sig_id}.mat", key)

def load_local(sig_id: str, key: str | None = None) -> np.ndarray:
    """
    Return trajectory array shape (T, 3) [x, y, p].
    """
    return _load_mat(L_DIR / f"{sig_id}.mat", key)