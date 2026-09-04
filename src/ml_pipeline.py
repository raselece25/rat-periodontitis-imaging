"""
ml_pipeline.py

A small machine-learning analysis step applied to the multimodal
(LSCI blood-flow + NIR oxygenation) feature set: classifies LPS-treated
vs. control animals from their peak-timepoint imaging features, as a
simple demonstration of the "Monte Carlo + DL analysis" style pipeline
described in the associated resume/project summary.

This is intentionally a small, transparent classifier (logistic
regression) rather than a deep network, since the feature space here is
low-dimensional (2 features x 1 peak timepoint per animal); the same
data-loading / evaluation scaffolding would apply to a larger model
trained on raw imaging time series.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler


def extract_peak_features(cohort: dict) -> tuple[np.ndarray, np.ndarray]:
    """Extract a simple per-animal feature vector: (max BFI, min oxy index)
    across the observed timepoints, i.e. the peak inflammatory response.
    """
    bfi_peak = cohort["bfi"].max(axis=1)
    oxy_trough = cohort["oxy_index"].min(axis=1)
    x = np.stack([bfi_peak, oxy_trough], axis=1)
    y = (cohort["group"] == "LPS").astype(int)
    return x, y


def cross_validated_accuracy(x: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = 0) -> dict:
    """Fit a logistic-regression classifier with stratified k-fold CV and
    report accuracy plus the cross-validated predictions (for plotting).
    """
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    clf = LogisticRegression()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_pred = cross_val_predict(clf, x_scaled, y, cv=cv)

    accuracy = float(np.mean(y_pred == y))
    return {"accuracy": accuracy, "y_true": y, "y_pred": y_pred}
