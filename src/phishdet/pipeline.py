"""End-to-end experiment: data -> features -> split -> models -> results/.

Run through the top-level script:  python run_pipeline.py [--external]
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedGroupKFold, cross_validate, train_test_split

from . import plots
from .data_gen import generate_dataset
from .evasion import generate_evasion_sets
from .features import FEATURE_NAMES, HOST_FEATURES, LEXICAL_FEATURES, extract_feature_frame, parse_url
from .metrics import compute_metrics
from .models import MODEL_LABELS, build_models

THRESHOLD = 0.5


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _rounded(obj, digits: int = 6):
    """Recursively convert numpy scalars to Python types and round floats."""
    if isinstance(obj, dict):
        return {k: _rounded(v, digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_rounded(v, digits) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return round(float(obj), digits)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def _grouped_split(y, groups, seed: int):
    """80/20 split in which no registered domain appears on both sides.

    A random split would put e.g. 'github.com/alice' in training and
    'github.com/bob' in test, letting the model score well by recognising
    domains it has already seen. Grouping by registered domain measures
    generalisation to *unseen* domains, which is what matters for new
    phishing campaigns. StratifiedGroupKFold keeps the class balance close to
    50/50; we use its first fold as the test set.
    """
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, test_idx = next(sgkf.split(np.zeros(len(y)), y, groups))
    return train_idx, test_idx


def run(out_dir: Path, data_dir: Path, seed: int = 42, n_per_class: int = 10_000,
        external: bool = False, n_external: int = 5000) -> dict:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"seed": seed, "n_per_class": n_per_class, "threshold": THRESHOLD}

    # ------------------------------------------------------------------ data
    _log(f"Generating synthetic corpus ({n_per_class} URLs per class, seed={seed})")
    df = generate_dataset(n_per_class=n_per_class, seed=seed)
    df["registered_domain"] = [parse_url(u).registered_domain for u in df.url]
    y = df.label.to_numpy()

    _log("Extracting features")
    X = extract_feature_frame(df.url)

    train_idx, test_idx = _grouped_split(y, df.registered_domain, seed)
    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"
    df.to_csv(data_dir / "synthetic_urls.csv", index=False)
    X_tr, X_te, y_tr, y_te = X.iloc[train_idx], X.iloc[test_idx], y[train_idx], y[test_idx]
    groups_tr = df.registered_domain.iloc[train_idx].to_numpy()
    assert not set(groups_tr) & set(df.registered_domain.iloc[test_idx]), "domain leakage"

    counts = (df.groupby(["split", "label", "technique"]).size().rename("n_urls").reset_index()
              .sort_values(["split", "label", "n_urls"], ascending=[False, True, False]))
    counts.to_csv(out_dir / "dataset_summary.csv", index=False)
    summary["dataset"] = {
        "n_total": len(df), "n_train": len(train_idx), "n_test": len(test_idx),
        "test_phishing_share": float(y_te.mean()),
        "n_registered_domains": int(df.registered_domain.nunique()),
        "n_features": len(FEATURE_NAMES),
    }
    _log(f"  train={len(train_idx)}  test={len(test_idx)}  test phishing share={y_te.mean():.3f}")

    # --------------------------------------------------- cross-validation
    _log("5-fold grouped cross-validation on the training set")
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_rows = []
    for key, model in build_models(seed).items():
        res = cross_validate(model, X_tr, y_tr, groups=groups_tr, cv=cv,
                             scoring=["roc_auc", "f1", "accuracy"], n_jobs=1)
        cv_rows.append({"model": MODEL_LABELS[key],
                        **{f"cv_{m}_mean": res[f"test_{m}"].mean() for m in ("accuracy", "f1", "roc_auc")},
                        **{f"cv_{m}_std": res[f"test_{m}"].std() for m in ("accuracy", "f1", "roc_auc")}})
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(out_dir / "cv_results.csv", index=False, float_format="%.4f")

    # ------------------------------------------------ fit + test evaluation
    _log("Fitting models on the full training set and scoring the test set")
    fitted, probas, rows, cms = {}, {}, [], {}
    for key, model in build_models(seed).items():
        model.fit(X_tr, y_tr)
        fitted[key] = model
        probas[key] = model.predict_proba(X_te)[:, 1]
        m = compute_metrics(y_te, probas[key], THRESHOLD)
        cms[key] = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
        rows.append({"model": MODEL_LABELS[key], **m})
    metrics_df = pd.DataFrame(rows).merge(cv_df, on="model")
    metrics_df.to_csv(out_dir / "metrics_comparison.csv", index=False, float_format="%.4f")
    pd.DataFrame([{"model": MODEL_LABELS[k], "actual": a, "predicted": p, "count": cms[k][i][j]}
                  for k in cms for i, a in enumerate(["legitimate", "phishing"])
                  for j, p in enumerate(["legitimate", "phishing"])]).to_csv(
        out_dir / "confusion_matrices.csv", index=False)
    summary["test_metrics"] = {r["model"]: {k: v for k, v in r.items() if k != "model"} for r in rows}
    best = max(fitted, key=lambda k: summary["test_metrics"][MODEL_LABELS[k]]["roc_auc"])
    summary["best_model_by_auc"] = MODEL_LABELS[best]
    print(metrics_df[["model", "accuracy", "precision", "recall", "f1", "roc_auc", "fpr"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # --------------------------------- comparison: naive random (leaky) split
    _log("Comparison run with a naive random split (domains may overlap)")
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    rand_rows = []
    for key, model in build_models(seed).items():
        model.fit(Xr_tr, yr_tr)
        rand_rows.append({"model": MODEL_LABELS[key],
                          **compute_metrics(yr_te, model.predict_proba(Xr_te)[:, 1], THRESHOLD)})
    pd.DataFrame(rand_rows).to_csv(out_dir / "metrics_random_split.csv", index=False, float_format="%.4f")

    # ------------------------------------------- per-technique detection rate
    test_df = df.iloc[test_idx]
    tech_rows = []
    for key in fitted:
        pred = probas[key] >= THRESHOLD
        for tech, idx in test_df.groupby("technique").indices.items():
            tech_rows.append({"model": key, "technique": tech, "n": len(idx),
                              "true_class": "legitimate" if tech.startswith("legit:") else "phishing",
                              # share flagged as phishing: recall for phishing
                              # techniques, false-positive rate for legit:* rows
                              "flagged_rate": float(pred[idx].mean())})
    tech_df = pd.DataFrame(tech_rows)
    tech_df.assign(model=tech_df.model.map(MODEL_LABELS)).to_csv(
        out_dir / "error_by_category.csv", index=False, float_format="%.4f")

    # ------------------------------------------------ permutation importance
    _log("Permutation importance on the test set (10 repeats, ROC-AUC)")
    imp_rows = []
    for key, model in fitted.items():
        pi = permutation_importance(model, X_te, y_te, scoring="roc_auc", n_repeats=10,
                                    random_state=seed, n_jobs=1)
        imp_rows += [{"model": key, "feature": f, "importance_mean": pi.importances_mean[i],
                      "importance_std": pi.importances_std[i]} for i, f in enumerate(FEATURE_NAMES)]
    imp_df = pd.DataFrame(imp_rows)
    imp_df.assign(model=imp_df.model.map(MODEL_LABELS)).sort_values(
        ["model", "importance_mean"], ascending=[True, False]).to_csv(
        out_dir / "feature_importance.csv", index=False, float_format="%.5f")
    coefs = fitted["logreg"][-1].coef_[0]
    pd.DataFrame({"feature": FEATURE_NAMES, "standardised_log1p_coefficient": coefs}).sort_values(
        "standardised_log1p_coefficient", key=np.abs, ascending=False).to_csv(
        out_dir / "logreg_coefficients.csv", index=False, float_format="%.4f")

    # ------------------------------------------------------------- ablation
    _log("Feature-group ablation (Gradient Boosting)")
    abl_rows = []
    for name, cols in [("Lexical only", LEXICAL_FEATURES), ("Host-based only", HOST_FEATURES),
                       ("All features", FEATURE_NAMES)]:
        model = clone(build_models(seed)["grad_boost"]).fit(X_tr[cols], y_tr)
        m = compute_metrics(y_te, model.predict_proba(X_te[cols])[:, 1], THRESHOLD)
        abl_rows.append({"feature_set": name, "n_features": len(cols), **m})
    abl_df = pd.DataFrame(abl_rows)
    abl_df.to_csv(out_dir / "ablation.csv", index=False, float_format="%.4f")

    # -------------------------------------------------------------- evasion
    _log("Adaptive-attacker (evasion) experiment")
    ev = generate_evasion_sets(n_per_level=2000, seed=seed + 1000)
    X_ev = extract_feature_frame(ev.url)
    ev_rows = []
    for key, model in fitted.items():
        p = model.predict_proba(X_ev)[:, 1]
        for (lvl, name), idx in ev.groupby(["level", "level_name"]).indices.items():
            ev_rows.append({"model": key, "level": lvl, "level_name": name, "n": len(idx),
                            "detection_rate": float((p[idx] >= THRESHOLD).mean()),
                            "mean_score": float(p[idx].mean())})
    ev_df = pd.DataFrame(ev_rows)
    ev_df.assign(model=ev_df.model.map(MODEL_LABELS)).to_csv(out_dir / "evasion.csv", index=False,
                                                             float_format="%.4f")
    fpr_ref = {k: summary["test_metrics"][MODEL_LABELS[k]]["fpr"] for k in fitted}

    # ------------------------------------------------- external real-world check
    if external:
        from .external import FEED_COMMIT, load_real_phishing_sample
        _log("External check on real phishing URLs (Phishing.Database)")
        real = load_real_phishing_sample(data_dir / "external", n=n_external, seed=seed)
        X_real = extract_feature_frame(real)
        ext_rows = [{"model": MODEL_LABELS[k], "n_urls": len(real),
                     "detection_rate": float((m.predict_proba(X_real)[:, 1] >= THRESHOLD).mean()),
                     "feed_commit": FEED_COMMIT} for k, m in fitted.items()]
        pd.DataFrame(ext_rows).to_csv(out_dir / "external_check.csv", index=False, float_format="%.4f")
        # Break real-URL detection down by whether *any* classic indicator fires.
        indicator = ((X_real[["has_ip_host", "has_at", "suspicious_tld", "brand_outside_official_domain",
                              "brand_lookalike", "is_punycode"]].sum(axis=1) > 0)
                     | (X_real.n_sensitive_words > 0)).to_numpy()
        ind_rows = []
        for k, m in fitted.items():
            flagged = m.predict_proba(X_real)[:, 1] >= THRESHOLD
            for name, mask in [("has classic indicator", indicator), ("no classic indicator", ~indicator)]:
                ind_rows.append({"model": MODEL_LABELS[k], "subset": name, "n_urls": int(mask.sum()),
                                 "share_of_sample": float(mask.mean()),
                                 "detection_rate": float(flagged[mask].mean())})
        pd.DataFrame(ind_rows).to_csv(out_dir / "external_check_by_indicator.csv", index=False,
                                      float_format="%.4f")
        # Same indicator on the synthetic corpus, for comparison.
        syn_ind = ((X[["has_ip_host", "has_at", "suspicious_tld", "brand_outside_official_domain",
                       "brand_lookalike", "is_punycode"]].sum(axis=1) > 0) | (X.n_sensitive_words > 0))
        summary["indicator_share"] = {"synthetic_phishing": float(syn_ind[y == 1].mean()),
                                      "synthetic_legitimate": float(syn_ind[y == 0].mean()),
                                      "real_phishing": float(indicator.mean())}
        shift = pd.DataFrame({"synthetic_legitimate": X[y == 0].mean(),
                              "synthetic_phishing": X[y == 1].mean(),
                              "real_phishing": X_real.mean()})
        shift.index.name = "feature"
        shift.to_csv(out_dir / "external_feature_shift.csv", float_format="%.4f")
        summary["external_check"] = {r["model"]: r["detection_rate"] for r in ext_rows}
        summary["external_check"]["n_urls"] = len(real)
        summary["external_check"]["feed_commit"] = FEED_COMMIT

    # -------------------------------------------------------------- figures
    _log("Rendering figures")
    plots.plot_roc(y_te, probas, fig_dir / "roc_curves.png")
    plots.plot_pr(y_te, probas, fig_dir / "pr_curves.png")
    plots.plot_confusion(cms, fig_dir / "confusion_matrices.png")
    plots.plot_importance(imp_df, best, fig_dir / "feature_importance.png")
    plots.plot_distributions(X, y, fig_dir / "feature_distributions.png")
    plots.plot_error_by_category(tech_df, fig_dir / "error_by_category.png")
    plots.plot_evasion(ev_df, fpr_ref, fig_dir / "evasion.png")
    plots.plot_ablation(abl_df, fig_dir / "ablation.png")

    with open(out_dir / "metrics.json", "w") as fh:
        # Rounded so parallel tree inference (last-bit float noise) cannot
        # change the file between runs.
        json.dump(_rounded(summary), fh, indent=2)
    _log(f"Done. Results written to {out_dir}/")
    return summary
