"""Model zoo: three classifiers of increasing capacity.

* Logistic regression -- linear, interpretable baseline. Features are on very
  different scales (URL length ~ 10^2, binary flags ~ 10^0), so it is wrapped in
  a StandardScaler pipeline. Count features are log1p-transformed to reduce
  the leverage of extreme lengths.
* Random forest -- bagged trees; captures interactions and thresholds without
  scaling, robust to noisy features.
* Gradient boosting (sklearn HistGradientBoostingClassifier) -- boosted trees,
  typically the strongest model on tabular data of this size.

Hyper-parameters are fixed, sensible defaults rather than tuned on the test
set; the cross-validation in pipeline.py only *reports* their stability.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

MODEL_LABELS = {
    "logreg": "Logistic Regression",
    "random_forest": "Random Forest",
    "grad_boost": "Gradient Boosting",
}


def build_models(seed: int = 42) -> dict:
    """Return fresh, unfitted estimators keyed by short model id."""
    return {
        "logreg": make_pipeline(
            FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
        ),
        "grad_boost": HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_iter=300,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=seed,
        ),
    }
