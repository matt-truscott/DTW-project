from __future__ import annotations

from pathlib import Path
from hashlib import blake2b
import json
from typing import Dict, List

import numpy as np
import pandas as pd

# ---- Protocol constants -----------------------------------------------------

REF_ATTEMPTS: List[int] = [1, 2, 6, 7]   # references in session 1 (genuine)
SEED_BASE: int = 25935569                # deterministic seed base

# ---- Internal helpers -------------------------------------------------------

def _project_root_from_proc(proc_root: Path) -> Path:
    """
    Given proc_root=<project_root>/data/processed, return <project_root>.
    """
    pr = Path(proc_root).resolve()
    return pr.parents[1]  # processed -> data -> <project_root>

def _stable_pair_id(*fields: object) -> int:
    """
    64-bit stable ID derived from the tuple of identifying fields.
    """
    key = "|".join(str(x) for x in fields)
    return int.from_bytes(blake2b(key.encode(), digest_size=8).digest(), "little", signed=False)

def _coerce_catalog_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure columns are plain Python types to avoid numpy/pandas Scalar complaints.
    """
    df = df.copy()
    df["user"]    = df["user"].astype(int)
    df["session"] = df["session"].astype(int)
    df["attempt"] = df["attempt"].astype(int)
    df["label"]   = df["label"].astype(str)
    df["path_lf"] = df["path_lf"].astype(str)
    return df

def _load_catalog(project_root: Path) -> pd.DataFrame:
    """
    Load canonical catalog created in 01_catalog:
    data/catalog/biosecurid_catalog.parquet
    Must contain: path_lf, user, session, attempt, label
    """
    cat_path = project_root / "data" / "catalog" / "biosecurid_catalog.parquet"
    if not cat_path.exists():
        raise FileNotFoundError(f"Missing catalog: {cat_path}. Run 01_catalog first.")
    df = pd.read_parquet(cat_path, engine="pyarrow")
    required = {"path_lf", "user", "session", "attempt", "label"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Catalog missing required columns: {missing}")
    return _coerce_catalog_types(df[["path_lf", "user", "session", "attempt", "label"]])

def _load_splits(project_root: Path) -> tuple[list[int], list[int]]:
    spath = project_root / "data" / "splits" / "user_splits.json"
    if not spath.exists():
        raise FileNotFoundError(f"Missing splits file: {spath}. Run 02_splits first.")
    payload = json.loads(spath.read_text())
    dev  = [int(u) for u in payload["dev_users"]]
    test = [int(u) for u in payload["test_users"]]
    return dev, test

def _refs_for_user(dfu: pd.DataFrame) -> Dict[int, str]:
    """
    Return {attempt -> path_lf} for the 4 genuine refs in session 1.
    (Use records to avoid pandas/numpy Scalar typing.)
    """
    refs = dfu[
        (dfu["session"] == 1) &
        (dfu["label"] == "genuine") &
        (dfu["attempt"].isin(REF_ATTEMPTS))
    ].sort_values("attempt")[["attempt", "path_lf"]]

    if len(refs) != 4:
        u = int(dfu["user"].iloc[0]) if len(dfu) else -1
        raise ValueError(f"Expected 4 refs in s0001 for user {u}; got {len(refs)}")

    mapping: Dict[int, str] = {}
    for rec in refs.to_dict(orient="records"):
        mapping[int(rec["attempt"])] = str(rec["path_lf"])
    return mapping

def _genuine_queries(dfu: pd.DataFrame) -> pd.DataFrame:
    q = dfu[(dfu["session"] != 1) & (dfu["label"] == "genuine")].sort_values(["session", "attempt"])
    if len(q) != 12:
        u = int(dfu["user"].iloc[0]) if len(dfu) else -1
        raise ValueError(f"User {u}: expected 12 genuine queries (sessions 2–4); got {len}(q)")
    return q[["session", "attempt", "path_lf"]]

def _skilled_queries(dfu: pd.DataFrame) -> pd.DataFrame:
    q = dfu[dfu["label"] == "skilled"].sort_values(["session", "attempt"])
    if len(q) != 12:
        u = int(dfu["user"].iloc[0]) if len(dfu) else -1
        raise ValueError(f"User {u}: expected 12 skilled queries; got {len(q)}")
    return q[["session", "attempt", "path_lf"]]

def _select_impostor_users(all_users_sorted: list[int], user: int, k: int = 12) -> list[int]:
    pool = [u for u in all_users_sorted if u != user]
    rng = np.random.default_rng(SEED_BASE + int(user))
    idx = rng.choice(len(pool), size=k, replace=False)
    return sorted(pool[i] for i in idx)

def _pick_impostor_attempt(user: int, impostor_user: int) -> int:
    """
    Deterministic mapping to one of {1,2,6,7} for the impostor's session-1 genuine attempts.
    """
    attempts = REF_ATTEMPTS
    idx = (SEED_BASE + user * 1315423911 + impostor_user * 2654435761) % 4
    return attempts[idx]

# ---- Core builders ----------------------------------------------------------

def _expand_pairs_for_case(
    df_catalog: pd.DataFrame,
    dfu: pd.DataFrame,
    users_sorted: list[int],
    user: int,
    case: str,
    split: str,
) -> list[dict]:
    """
    Produce comparison-level rows for one user and one case.
    Each row is one (ref_attempt × query) pair.
    """
    out: list[dict] = []
    ref_map = _refs_for_user(dfu)  # attempt -> path

    if case == "genuine":
        qdf = _genuine_queries(dfu)
        for rec in qdf.to_dict(orient="records"):
            q_sess = int(rec["session"])
            q_att  = int(rec["attempt"])
            q_path = str(rec["path_lf"])
            for ref_attempt in REF_ATTEMPTS:
                out.append({
                    "pair_id": _stable_pair_id(split, case, user, ref_attempt, user, q_sess, q_att),
                    "user": int(user),
                    "split": split,
                    "case": case,
                    "label": 1,
                    "ref_session": 1,
                    "ref_attempt": int(ref_attempt),
                    "query_user": int(user),
                    "query_session": q_sess,
                    "query_attempt": q_att,
                    "path_ref": ref_map[int(ref_attempt)],
                    "path_query": q_path,
                })

    elif case == "skilled":
        qdf = _skilled_queries(dfu)
        for rec in qdf.to_dict(orient="records"):
            q_sess = int(rec["session"])
            q_att  = int(rec["attempt"])
            q_path = str(rec["path_lf"])
            for ref_attempt in REF_ATTEMPTS:
                out.append({
                    "pair_id": _stable_pair_id(split, case, user, ref_attempt, user, q_sess, q_att),
                    "user": int(user),
                    "split": split,
                    "case": case,
                    "label": 0,
                    "ref_session": 1,
                    "ref_attempt": int(ref_attempt),
                    "query_user": int(user),
                    "query_session": q_sess,
                    "query_attempt": q_att,
                    "path_ref": ref_map[int(ref_attempt)],
                    "path_query": q_path,
                })

    elif case == "random":
        impostors = _select_impostor_users(users_sorted, user, k=12)
        for v in impostors:
            att = _pick_impostor_attempt(user, v)
            qrow = df_catalog[
                (df_catalog["user"] == v) &
                (df_catalog["session"] == 1) &
                (df_catalog["label"] == "genuine") &
                (df_catalog["attempt"] == att)
            ]
            if qrow.empty:
                qrow = (df_catalog[
                    (df_catalog["user"] == v) &
                    (df_catalog["session"] == 1) &
                    (df_catalog["label"] == "genuine")
                ].sort_values("attempt").head(1))
            if qrow.empty:
                raise ValueError(f"Impostor pick failed for user {v} (no session-1 genuine attempts).")

            rec = qrow[["session", "attempt", "path_lf"]].iloc[0].to_dict()
            q_sess = int(rec["session"])
            q_att  = int(rec["attempt"])
            q_path = str(rec["path_lf"])

            for ref_attempt in REF_ATTEMPTS:
                out.append({
                    "pair_id": _stable_pair_id(split, case, user, ref_attempt, int(v), q_sess, q_att),
                    "user": int(user),
                    "split": split,
                    "case": case,
                    "label": 0,
                    "ref_session": 1,
                    "ref_attempt": int(ref_attempt),
                    "query_user": int(v),
                    "query_session": q_sess,
                    "query_attempt": q_att,
                    "path_ref": ref_map[int(ref_attempt)],
                    "path_query": q_path,
                })
    else:
        raise ValueError(f"Unknown case: {case}")

    return out

def _build_pairs_for_split(df_catalog: pd.DataFrame, users: list[int], split: str) -> dict[str, pd.DataFrame]:
    users_sorted = sorted(int(u) for u in df_catalog["user"].unique().tolist())
    by_user: Dict[int, pd.DataFrame] = {int(u): df_catalog[df_catalog["user"] == int(u)].copy() for u in users}

    out_by_case: dict[str, list[dict]] = {"genuine": [], "skilled": [], "random": []}

    for u in users:
        dfu = by_user[int(u)]
        for case in ("genuine", "skilled", "random"):
            out_by_case[case].extend(_expand_pairs_for_case(df_catalog, dfu, users_sorted, int(u), case, split))

    dfs = {case: pd.DataFrame(rows) for case, rows in out_by_case.items()}

    # Casting and ordering
    for case, dfp in dfs.items():
        if dfp.empty:
            continue
        int_cols = ["pair_id", "user", "label", "ref_session", "ref_attempt",
                    "query_user", "query_session", "query_attempt"]
        for c in int_cols:
            dfp[c] = dfp[c].astype("int64")
        dfp["split"] = dfp["split"].astype("string")
        dfp["case"]  = dfp["case"].astype("string")
        dfs[case] = dfp.sort_values(
            ["user", "case", "query_user", "query_session", "query_attempt", "ref_attempt"]
        ).reset_index(drop=True)

    return dfs

# ---- Public API -------------------------------------------------------------

def write_pairs_for_splits(
    proc_root: Path,
    *,
    out_dir: Path | None = None,
) -> dict[str, dict[str, Path]]:
    """
    Build comparison-level pairs for dev/test and write per-case Parquet files.

    Output files:
      data/pairs/<split>/pairs_<split>_{genuine|skilled|random}.parquet

    Returns: {split: {case: path}}
    """
    proc_root = Path(proc_root)
    project_root = _project_root_from_proc(proc_root)

    df_catalog = _load_catalog(project_root)
    dev_users, test_users = _load_splits(project_root)

    outputs: dict[str, dict[str, Path]] = {"dev": {}, "test": {}}

    for split, users in (("dev", dev_users), ("test", test_users)):
        dfs = _build_pairs_for_split(df_catalog, users, split)

        # Integrity: exactly 48 rows per user per case
        n_users = len(users)
        for case, dfp in dfs.items():
            expected = 48 * n_users
            got = len(dfp)
            if got != expected:
                raise AssertionError(f"{split}/{case}: expected {expected} rows, got {got}")

        # Write
        base_dir = (out_dir or (project_root / "data" / "pairs")) / split
        base_dir.mkdir(parents=True, exist_ok=True)
        for case, dfp in dfs.items():
            out_path = base_dir / f"pairs_{split}_{case}.parquet"
            dfp.to_parquet(out_path, index=False, engine="pyarrow")
            outputs[split][case] = out_path

    return outputs
