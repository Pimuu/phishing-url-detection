"""Evaluation metrics, including base-rate-adjusted precision."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Prevalence of phishing assumed for the "deployment" precision estimate.
# Real traffic is overwhelmingly benign; 1% is already a pessimistic
# (phishing-heavy) setting for e.g. URLs seen in corporate e-mail.
DEPLOYMENT_PREVALENCE = 0.01


def precision_at_prevalence(tpr: float, fpr: float, prevalence: float = DEPLOYMENT_PREVALENCE) -> float:
    """Precision the classifier would have if phishing made up `prevalence` of traffic.

    The test set is balanced (50% phishing), which flatters precision. By
    Bayes' rule, precision = TPR*pi / (TPR*pi + FPR*(1-pi)).
    """
    denom = tpr * prevalence + fpr * (1 - prevalence)
    return float(tpr * prevalence / denom) if denom > 0 else float("nan")


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    """Standard binary metrics at a fixed decision threshold, plus derived rates."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "avg_precision": average_precision_score(y_true, proba),
        "fpr": fpr,
        # Detection rate when the threshold is set for a 1% false-positive rate.
        "tpr_at_1pct_fpr": float(np.interp(0.01, *roc_curve(y_true, proba)[:2])),
        "precision_at_1pct_prevalence": precision_at_prevalence(tpr, fpr),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
