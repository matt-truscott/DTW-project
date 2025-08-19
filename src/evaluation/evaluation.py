import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

# ---------------------------------------------------------------------
# Back-compat (kept from your original)
# ---------------------------------------------------------------------

GENUINE_SET = {1, 2, 6, 7}
SKILLED_SET = {3, 4, 5}

def _ensure_query_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee a 'query_label' column in the merged pairs+cache DataFrame.
    Priority:
      1) pass-through if already present
      2) rename from query_label_pairs/_cache if present
      3) reconstruct from (user, query_user, query_attempt)
         - query_user != user  -> 'random'
         - attempt in GENUINE_SET -> 'genuine'
         - attempt in SKILLED_SET -> 'skilled'
    """
    if "query_label" in df.columns:
        return df

    for c in ("query_label_pairs", "query_label_cache"):
        if c in df.columns:
            df = df.rename(columns={c: "query_label"})
            return df

    # Reconstruct if we have enough info
    needed = {"user", "query_user", "query_attempt"}
    if needed.issubset(df.columns):
        def _infer(row):
            if row["query_user"] != row["user"]:
                return "random"
            att = int(row["query_attempt"])
            if att in GENUINE_SET:
                return "genuine"
            if att in SKILLED_SET:
                return "skilled"
            raise ValueError(f"Unknown attempt id {att} for pair_id={row.get('pair_id')}")
        df = df.copy()
        df["query_label"] = df.apply(_infer, axis=1)
        return df

    # Last resort: clear error with context
    raise KeyError(
        "No 'query_label' and not enough columns to reconstruct it. "
        "Expected one of: ['query_label', 'query_label_pairs', 'query_label_cache'] "
        "or the trio ['user','query_user','query_attempt']."
    )

def load_results(pairs_path: str | Path, cache_path: str | Path) -> pd.DataFrame:
    """
    Legacy loader for pairwise (pathA/pathB) experiments.
    """
    pairs = pd.read_parquet(pairs_path)
    if "pair_id" not in pairs.columns:
        pairs = pairs.reset_index(drop=False).rename(columns={"index": "pair_id"})
    cache = pd.read_parquet(cache_path)
    df = pairs.merge(cache, on="pair_id", how="inner", suffixes=("_pairs", "_cache"))
    # collapse duplicate label columns if both exist
    if "label_pairs" in df.columns and "label_cache" in df.columns:
        df["label"] = df["label_pairs"]; df.drop(["label_pairs","label_cache"], axis=1, inplace=True)
    elif "label_pairs" in df.columns:
        df.rename(columns={"label_pairs":"label"}, inplace=True)
    elif "label_cache" in df.columns:
        df.rename(columns={"label_cache":"label"}, inplace=True)
    return df


def compute_metrics(df: pd.DataFrame, score_col: str = "d_raw", label_col: str = "label") -> dict:
    """
    Legacy: higher score ⇒ more genuine.
    Returns ROC arrays, AUC, EER and its threshold (all threshold-free except EER threshold).
    """
    y_true  = df[label_col].to_numpy(dtype=int)
    y_score = df[score_col].to_numpy(dtype=float)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc = float(roc_auc_score(y_true, y_score))
    fnr = 1.0 - tpr
    idx = int(np.argmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    eer_threshold = float(thresholds[idx])
    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "auc": auc, "eer": eer, "eer_threshold": eer_threshold}


def plot_roc(fpr: np.ndarray, tpr: np.ndarray, auc: float, ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.legend(loc="lower right")
    return ax


def plot_det(fpr: np.ndarray, fnr: np.ndarray, ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(fpr, fnr, label="DET")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("False Negative Rate"); ax.legend()
    return ax


def save_metrics(metrics: dict, out_path: str | Path):
    """
    Save scalar metrics as JSON or CSV based on extension.
    Accepts keys: auc, eer, eer_threshold (case-insensitive).
    """
    out_path = Path(out_path)
    def get(key: str):
        for k, v in metrics.items():
            if k.lower() == key.lower():
                return v
        raise KeyError(f"Metric '{key}' not found in {list(metrics.keys())}")
    data = {"auc": get("auc"), "eer": get("eer"), "eer_threshold": get("eer_threshold")}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".json":
        out_path.write_text(json.dumps(data, indent=2))
    else:
        pd.DataFrame({"metric": list(data.keys()), "value": list(data.values())}).to_csv(out_path, index=False)

# ---------------------------------------------------------------------
# New: evaluation for query→refs cache (d_mean / d_min) and scenarios
# ---------------------------------------------------------------------

SCENARIOS = ("skilled_only", "random_only", "skilled_random")

def _load_q2refs_split(proc_root: Path, split: str, score_agg: str = "d_mean") -> pd.DataFrame:
    proc_root = Path(proc_root)
    pairs = pd.read_parquet(proc_root / f"pairs_{split}.parquet", engine="pyarrow")
    cache = pd.read_parquet(proc_root / f"dtw_cache_{split}.parquet", engine="pyarrow")

    df = pairs.merge(cache, on="pair_id", how="inner", suffixes=("_pairs", "_cache"))

    # NEW: ensure we always have query_label available downstream
    df = _ensure_query_label(df)

    if score_agg not in df.columns:
        raise KeyError(f"'{score_agg}' not found in cache columns: {sorted(df.columns)}")
    df["score"] = -df[score_agg].astype(float)   # higher = more genuine
    return df


def _make_binary(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """
    Map 'query_label' to binary {genuine=1, forgery=0} for a scenario,
    and filter rows to the subset needed.
    """
    allowed = {"genuine", "skilled", "random"}
    if scenario == "skilled_only":
        sub = df[df["query_label"].isin(["genuine", "skilled"])].copy()
        sub["label"] = (sub["query_label"] == "genuine").astype(int)
    elif scenario == "random_only":
        sub = df[df["query_label"].isin(["genuine", "random"])].copy()
        sub["label"] = (sub["query_label"] == "genuine").astype(int)
    elif scenario == "skilled_random":
        sub = df[df["query_label"].isin(list(allowed))].copy()
        sub["label"] = (sub["query_label"] == "genuine").astype(int)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    return sub


def _min_eer_threshold(fpr: np.ndarray, tpr: np.ndarray, thr: np.ndarray) -> Tuple[float, float]:
    fnr = 1.0 - tpr
    idx = int(np.argmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return float(thr[idx]), eer


def _tpr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, targets=(0.01, 0.05)) -> Dict[str, float]:
    out = {}
    for x in targets:
        # interpolate TPR at target FPR
        out[f"TPR@FPR={x:.2f}"] = float(np.interp(x, fpr, tpr))
    return out


@dataclass
class EvalResult:
    scenario: str
    score_agg: str
    auc_dev: float
    eer_dev: float
    thr_dev: float
    auc_test: float
    eer_test: float
    apcer_test: float   # = FPR at dev-threshold
    bpcer_test: float   # = FNR at dev-threshold
    tpr_at_001_test: float
    tpr_at_005_test: float


def evaluate_scenario(proc_root: Path, scenario: str, *, score_agg: str = "d_mean") -> Tuple[EvalResult, dict, dict]:
    """
    Full evaluation for one scenario:
      - load dev/test merges
      - compute dev ROC; choose threshold at min EER
      - compute test ROC/AUC/EER
      - compute APCER/BPCER on test at dev threshold + TPR@FPR {0.01, 0.05} (threshold-free)
    Returns (EvalResult, curves_dev, curves_test) where curves_* contain fpr/tpr/thresholds.
    """
    # Load and filter
    df_dev  = _make_binary(_load_q2refs_split(proc_root, "dev",  score_agg), scenario)
    df_test = _make_binary(_load_q2refs_split(proc_root, "test", score_agg), scenario)

    # Dev ROC + threshold
    fpr_d, tpr_d, thr_d, auc_d = _roc_arrays(df_dev["label"].to_numpy(int), df_dev["score"].to_numpy(float))
    thr_dev, eer_dev = _min_eer_threshold(fpr_d, tpr_d, thr_d)

    # Test ROC + EER
    fpr_t, tpr_t, thr_t, auc_t = _roc_arrays(df_test["label"].to_numpy(int), df_test["score"].to_numpy(float))
    _, eer_test = _min_eer_threshold(fpr_t, tpr_t, thr_t)

    # APCER/BPCER at dev threshold on test
    y_true = df_test["label"].to_numpy(int)
    y_score = df_test["score"].to_numpy(float)
    y_pred = (y_score >= thr_dev).astype(int)  # positive= genuine

    # Confusion-derived rates
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    # In presentation-attack terms:
    apcer = fp / max(fp + tn, 1)  # false accept rate for attacks (= FPR)
    bpcer = fn / max(fn + tp, 1)  # bona fide false reject rate (= FNR)

    # Threshold-free TPR@FPR on test
    tprs = _tpr_at_fpr(fpr_t, tpr_t, targets=(0.01, 0.05))

    res = EvalResult(
        scenario=scenario,
        score_agg=score_agg,
        auc_dev=auc_d,
        eer_dev=eer_dev,
        thr_dev=thr_dev,
        auc_test=auc_t,
        eer_test=eer_test,
        apcer_test=float(apcer),
        bpcer_test=float(bpcer),
        tpr_at_001_test=float(tprs["TPR@FPR=0.01"]),
        tpr_at_005_test=float(tprs["TPR@FPR=0.05"]),
    )

    curves_dev  = {"fpr": fpr_d, "tpr": tpr_d, "thresholds": thr_d}
    curves_test = {"fpr": fpr_t, "tpr": tpr_t, "thresholds": thr_t}
    return res, curves_dev, curves_test

def _dtw_distance_np(a: np.ndarray, b: np.ndarray) -> float:
    # a, b: (T, F)
    N, M = len(a), len(b)
    D = np.full((N+1, M+1), np.inf, dtype=float)
    D[0, 0] = 0.0
    for i in range(1, N+1):
        ai = a[i-1]
        for j in range(1, M+1):
            cost = np.linalg.norm(ai - b[j-1])
            D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
    return float(D[N, M])

def _merge_case(project_root: Path, split: str, case: str, *, score_from: str = "d_raw") -> pd.DataFrame:
    """
    Merge pairs_<split>_<case>.parquet with dtw_cache_<split>_<case>.parquet on pair_id.
    Returns columns: [pair_id, label, score] (+ optional bookkeeping).
    score = -d_raw (or -d_bound) so that higher = more genuine.
    """
    project_root = Path(project_root)
    pairs = pd.read_parquet(project_root / "data" / "pairs" / split / f"pairs_{split}_{case}.parquet", engine="pyarrow")
    cache = pd.read_parquet(project_root / "data" / "dtw_cache" / split / f"dtw_cache_{split}_{case}.parquet", engine="pyarrow")

    df = pairs.merge(cache, on="pair_id", how="inner", suffixes=("_pairs", "_cache"))

    # prefer label from cache if present, else from pairs
    if "label_cache" in df.columns:
        df["label"] = df["label_cache"].astype(int)
    elif "label_pairs" in df.columns:
        df["label"] = df["label_pairs"].astype(int)
    else:
        df["label"] = df["label"].astype(int)

    if score_from not in ("d_raw", "d_bound"):
        raise ValueError("score_from must be 'd_raw' or 'd_bound'")
    if score_from not in df.columns:
        raise KeyError(f"Column '{score_from}' not found in merged DataFrame.")

    df["score"] = -df[score_from].astype(float)  # higher = more genuine
    return df[["pair_id", "label", "score"]].copy()


def _scenario_df(project_root: Path, split: str, scenario: str, *, score_from: str = "d_raw",
                 balance_impostors: bool = False, rng_seed: int = 25935569) -> pd.DataFrame:
    """
    Build the evaluation table for a split & scenario from per-case merges.
    Scenarios:
      - skilled_only  : genuine vs skilled
      - random_only   : genuine vs random
      - skilled_random: genuine vs (skilled ∪ random) [optionally balanced]
    """
    g = _merge_case(project_root, split, "genuine", score_from=score_from).assign(query_label="genuine")
    s = _merge_case(project_root, split, "skilled", score_from=score_from).assign(query_label="skilled")
    r = _merge_case(project_root, split, "random",  score_from=score_from).assign(query_label="random")

    if scenario == "skilled_only":
        df = pd.concat([g, s], ignore_index=True)
    elif scenario == "random_only":
        df = pd.concat([g, r], ignore_index=True)
    elif scenario == "skilled_random":
        impostors = pd.concat([s, r], ignore_index=True)
        if balance_impostors:
            # downsample impostors to match #genuine for the split
            n_g = len(g)
            if len(impostors) > n_g:
                rng = np.random.default_rng(rng_seed)
                idx = rng.choice(len(impostors), size=n_g, replace=False)
                impostors = impostors.iloc[idx]
        df = pd.concat([g, impostors], ignore_index=True)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    # Binary label already set (1=genuine, 0=impostor) from files.
    return df[["pair_id", "label", "score"]].copy()


def _roc_arrays(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    fpr, tpr, thr = roc_curve(y_true, y_score)
    auc = float(roc_auc_score(y_true, y_score))
    return fpr, tpr, thr, auc


def _eer_from_curve(fpr: np.ndarray, tpr: np.ndarray, thr: np.ndarray) -> Tuple[float, float]:
    fnr = 1.0 - tpr
    idx = int(np.argmin(np.abs(fnr - fpr)))
    return float(thr[idx]), float((fpr[idx] + fnr[idx]) / 2.0)


def _apcer_bpcer_at_thr(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> Tuple[float, float]:
    y_pred = (y_score >= thr).astype(int)  # positive=genuine
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    apcer = fp / max(fp + tn, 1)  # attack false accept rate (FPR)
    bpcer = fn / max(fn + tp, 1)  # bona fide false reject rate (FNR)
    return float(apcer), float(bpcer)


@dataclass
class ScenarioReport:
    scenario: str
    score_from: str
    auc_dev: float
    eer_dev: float
    thr_dev: float
    auc_test: float
    eer_test: float
    apcer_test: float
    bpcer_test: float


def evaluate_scenario_protocol(project_root: Path, scenario: str, *,
                               score_from: str = "d_raw",
                               balance_impostors_dev: bool = False) -> Tuple[ScenarioReport, Dict, Dict]:
    """
    Protocol-aligned evaluation:
      - Build dev/test tables from per-case merges
      - Choose threshold on dev at min-EER
      - Report AUC/EER on test and APCER/BPCER on test at the dev threshold
    """
    # Build dev/test
    dev  = _scenario_df(project_root, "dev",  scenario, score_from=score_from,
                        balance_impostors=balance_impostors_dev)
    test = _scenario_df(project_root, "test", scenario, score_from=score_from,
                        balance_impostors=False)

    # Dev ROC + threshold
    fpr_d, tpr_d, thr_d, auc_d = _roc_arrays(dev["label"].to_numpy(int), dev["score"].to_numpy(float))
    thr_dev, eer_dev = _eer_from_curve(fpr_d, tpr_d, thr_d)

    # Test ROC + EER
    fpr_t, tpr_t, thr_t, auc_t = _roc_arrays(test["label"].to_numpy(int), test["score"].to_numpy(float))
    _, eer_test = _eer_from_curve(fpr_t, tpr_t, thr_t)

    # APCER/BPCER at dev threshold (on test)
    apcer, bpcer = _apcer_bpcer_at_thr(test["label"].to_numpy(int), test["score"].to_numpy(float), thr_dev)

    rpt = ScenarioReport(
        scenario=scenario,
        score_from=score_from,
        auc_dev=auc_d,
        eer_dev=eer_dev,
        thr_dev=thr_dev,
        auc_test=auc_t,
        eer_test=eer_test,
        apcer_test=apcer,
        bpcer_test=bpcer,
    )
    curves_dev  = {"fpr": fpr_d, "tpr": tpr_d, "thresholds": thr_d}
    curves_test = {"fpr": fpr_t, "tpr": tpr_t, "thresholds": thr_t}
    return rpt, curves_dev, curves_test
