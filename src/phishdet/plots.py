"""All figures for the results/ folder (matplotlib only, headless backend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed; must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from .models import MODEL_LABELS

COLORS = {"logreg": "#4C72B0", "random_forest": "#55A868", "grad_boost": "#C44E52"}
CLASS_COLORS = {0: "#4C72B0", 1: "#DD8452"}
CLASS_NAMES = {0: "Legitimate", 1: "Phishing"}
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_roc(y_true, probas: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for key, proba in probas.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        ax.plot(fpr, tpr, color=COLORS[key], lw=2, label=f"{MODEL_LABELS[key]} (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="Chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="ROC curves (domain-disjoint test set)", xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(loc="lower right")
    # Inset: the low-FPR region that matters operationally.
    inset = ax.inset_axes([0.42, 0.32, 0.36, 0.36])
    for key, proba in probas.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        inset.plot(fpr, tpr, color=COLORS[key], lw=1.5)
    inset.set(xlim=(0, 0.1), ylim=(0.5, 1.0), title="FPR ≤ 10%")
    inset.title.set_fontsize(8)
    inset.tick_params(labelsize=7)
    _save(fig, path)


def plot_pr(y_true, probas: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for key, proba in probas.items():
        prec, rec, _ = precision_recall_curve(y_true, proba)
        ax.plot(rec, prec, color=COLORS[key], lw=2, label=MODEL_LABELS[key])
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves (balanced test set)",
           xlim=(0, 1), ylim=(0.4, 1.01))
    ax.legend(loc="lower left")
    _save(fig, path)


def plot_confusion(cms: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, len(cms), figsize=(4.2 * len(cms), 3.8))
    for ax, (key, cm) in zip(np.atleast_1d(axes), cms.items()):
        cm = np.asarray(cm)
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.sum(axis=1).max())
        for i in range(2):
            for j in range(2):
                pct = cm[i, j] / cm[i].sum() * 100
                ax.text(j, i, f"{cm[i, j]}\n({pct:.1f}%)", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Legit", "Phish"], yticklabels=["Legit", "Phish"],
               xlabel="Predicted", ylabel="Actual", title=MODEL_LABELS[key])
        ax.spines[:].set_visible(False)
    _save(fig, path)


def plot_importance(imp: pd.DataFrame, order_by: str, path: Path, top: int = 15) -> None:
    """Grouped horizontal bars of permutation importance (drop in ROC-AUC)."""
    top_feats = imp[imp.model == order_by].nlargest(top, "importance_mean").feature.tolist()[::-1]
    models = list(dict.fromkeys(imp.model))
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * top + 1.2))
    height = 0.8 / len(models)
    y = np.arange(len(top_feats))
    for i, key in enumerate(models):
        sub = imp[imp.model == key].set_index("feature").reindex(top_feats)
        ax.barh(y + (i - (len(models) - 1) / 2) * height, sub.importance_mean, height=height,
                xerr=sub.importance_std, color=COLORS[key], label=MODEL_LABELS[key], error_kw={"lw": 0.8})
    ax.set_yticks(y, top_feats)
    ax.axvline(0, color="black", lw=0.8)
    ax.set(xlabel="Permutation importance (mean drop in test ROC-AUC, 10 repeats)",
           title=f"Top-{top} features (ordered by {MODEL_LABELS[order_by]})")
    ax.legend(loc="lower right")
    _save(fig, path)


def plot_distributions(X: pd.DataFrame, y, path: Path) -> None:
    """Class-conditional distributions of key features."""
    y = np.asarray(y)
    continuous = ["url_length", "hostname_length", "n_digits", "domain_entropy", "n_dots", "path_depth"]
    binary = ["is_https", "has_ip_host", "has_at", "suspicious_tld", "brand_outside_official_domain",
              "brand_lookalike", "n_sensitive_words", "is_free_hosting"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax, feat in zip(axes.flat[:6], continuous):
        values = X[feat]
        hi = np.percentile(values, 99)
        bins = np.arange(0, hi + 2) - 0.5 if feat in ("n_dots", "path_depth") else np.linspace(values.min(), hi, 40)
        for label in (0, 1):
            ax.hist(values[y == label].clip(upper=hi), bins=bins, alpha=0.55, density=True,
                    color=CLASS_COLORS[label], label=CLASS_NAMES[label])
        ax.set(title=feat, ylabel="density")
    axes.flat[0].legend()
    # Binary / small-count features: share of URLs where the feature is > 0.
    for ax, feats in zip(axes.flat[6:], (binary[:4], binary[4:])):
        x = np.arange(len(feats))
        for k, label in enumerate((0, 1)):
            rates = [(X.loc[y == label, f] > 0).mean() * 100 for f in feats]
            ax.bar(x + (k - 0.5) * 0.4, rates, width=0.4, color=CLASS_COLORS[label], label=CLASS_NAMES[label])
        ax.set_xticks(x, [f.replace("brand_outside_official_domain", "brand_outside\n_official").replace("n_sensitive_words", "sensitive\nword") for f in feats],
                      rotation=20, fontsize=8)
        ax.set(ylabel="% of URLs with feature > 0", ylim=(0, 100), title="Binary indicators")
    fig.suptitle("Feature distributions: phishing vs legitimate (full synthetic corpus)", y=1.0)
    _save(fig, path)


def plot_error_by_category(df: pd.DataFrame, path: Path) -> None:
    """Left: detection rate per phishing technique. Right: FPR per legit category."""
    order = [k for k in MODEL_LABELS if k in set(df.model)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    panels = [("phishing", "Detection rate (%) -- higher is better", "Phishing: detection rate by technique"),
              ("legitimate", "Flagged as phishing (%) -- lower is better", "Legitimate: false-positive rate by category")]
    for ax, (cls, xlabel, title) in zip(axes, panels):
        sub = df[df.true_class == cls]
        pivot = sub.pivot(index="technique", columns="model", values="flagged_rate")[order] * 100
        pivot.index = [t.replace("legit:", "") for t in pivot.index]
        pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=(cls == "phishing")).index]
        n = sub.drop_duplicates("technique").set_index("technique").n
        n.index = [t.replace("legit:", "") for t in n.index]
        y = np.arange(len(pivot))
        h = 0.8 / len(order)
        for i, key in enumerate(order):
            ax.barh(y + (len(order) - 1 - 2 * i) * h / 2, pivot[key], height=h,
                    color=COLORS[key], label=MODEL_LABELS[key])
        ax.set_yticks(y, [f"{t} (n={n[t]})" for t in pivot.index])
        ax.set(xlabel=xlabel, title=title, xlim=(0, 100))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04), frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_evasion(df: pd.DataFrame, fpr_ref: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    levels = df.level_name.unique()
    x = np.arange(len(levels))
    for key in df.model.unique():
        sub = df[df.model == key].set_index("level_name").reindex(levels)
        ax.plot(x, sub.detection_rate * 100, marker="o", lw=2, color=COLORS[key], label=MODEL_LABELS[key])
        ax.axhline(fpr_ref[key] * 100, color=COLORS[key], ls=":", lw=1)
    ax.set_xticks(x, [lv.replace(" + ", "\n+ ") for lv in levels], fontsize=8)
    ax.set(ylabel="Phishing URLs detected (%)", ylim=(0, 102),
           title="Adaptive attacker: detection rate vs. evasion level\n(dotted lines: false-positive rate on legitimate test URLs)")
    ax.legend(loc="lower left")
    _save(fig, path)


def plot_ablation(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df.roc_auc, width=0.4, label="ROC-AUC", color="#4C72B0")
    ax.bar(x + 0.2, df.f1, width=0.4, label="F1", color="#DD8452")
    for i, row in enumerate(df.itertuples()):
        ax.text(i - 0.2, row.roc_auc + 0.003, f"{row.roc_auc:.3f}", ha="center", fontsize=8)
        ax.text(i + 0.2, row.f1 + 0.003, f"{row.f1:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, df.feature_set)
    lo = min(df.f1.min(), df.roc_auc.min())
    ax.set(ylim=(max(0, lo - 0.08), 1.02), ylabel="Score",
           title="Feature-group ablation (Gradient Boosting)")
    ax.legend(loc="lower right")
    _save(fig, path)
