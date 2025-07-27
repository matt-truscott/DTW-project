"""
Calibration utilities: simple normalization of DTW scores and
learning a bounded [0,1] mapping via a small classifier.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

def add_normalizations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with columns
      ['d_raw','path_len','len_ref','len_qry',…]
    compute various normalized distances and return an enriched copy:
      - d_by_path:     d_raw / path_len
      - d_by_ref_len:  d_raw / len_ref
      - d_by_qry_len:  d_raw / len_qry
      - d_by_avg_len:  d_raw / ((len_ref + len_qry)/2)
    """
    df2 = df.copy()
    df2['d_by_path']    = df2['d_raw'] / df2['path_len']
    df2['d_by_ref_len'] = df2['d_raw'] / df2['len_ref']
    df2['d_by_qry_len'] = df2['d_raw'] / df2['len_qry']
    df2['d_by_avg_len'] = df2['d_raw'] / ((df2['len_ref'] + df2['len_qry']) / 2)
    return df2

def train_calibrator(
    df: pd.DataFrame,
    feature_col: str,
    label_col: str = 'label',
    method: str = 'logistic',
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Train a small classifier to map one normalized DTW feature
    to a probability in [0,1].

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `feature_col` (float) and `label_col` (0/1).
    feature_col : str
        Column name to use as the single input feature.
    method : {'logistic','gbm'}
        Which model to train.
    test_size : float
        Fraction of data held out for testing.
    random_state : int

    Returns
    -------
    model : fitted classifier (has predict_proba)
    X_test : np.ndarray shape (n_test,1)
    y_test : np.ndarray shape (n_test,)
    y_score_test : np.ndarray shape (n_test,)  # predicted prob of class=1
    """
    X = df[[feature_col]].values
    y = df[label_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    if method == 'logistic':
        model = LogisticRegression(solver='liblinear', random_state=random_state)
    elif method == 'gbm':
        model = GradientBoostingClassifier(random_state=random_state)
    else:
        raise ValueError(f"Unknown method: {method!r}")

    model.fit(X_train, y_train)
    y_score_test = model.predict_proba(X_test)[:, 1]

    return model, X_test, y_test, y_score_test
