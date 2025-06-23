from pathlib import Path
import numpy as np
import pandas as pd

# Absolute root (change if you move the project)
RAW_ROOT = Path(r"C:\Users\mattt\Skripsie\Projects\DTW-project\data\raw")
G_DIR    = RAW_ROOT / "GlobalFeatures"
L_DIR    = RAW_ROOT / "LocalFunctions"

def list_ids():
    """Return all signature IDs without `.txt` extension, sorted."""
    return sorted(p.stem for p in G_DIR.glob("*.txt"))

def load_global(sig_id: str) -> np.ndarray:
    """1-D array of 49 global features."""
    return np.loadtxt(G_DIR / f"{sig_id}.txt", delimiter=",", dtype=float)

def load_local(sig_id: str) -> np.ndarray:
    """T × 3 array (x, y, p) for a variable-length stroke."""
    return np.loadtxt(L_DIR / f"{sig_id}.txt", delimiter=",", dtype=float)

def to_dataframe(sig_ids=None) -> pd.DataFrame:
    """Convenience: stack many global vectors into a DataFrame."""
    sig_ids = sig_ids or list_ids()
    data = [load_global(s) for s in sig_ids]
    return pd.DataFrame(data, index=sig_ids)