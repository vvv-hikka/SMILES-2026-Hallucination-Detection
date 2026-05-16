"""
probe.py — Hallucination probe classifier (student-implemented).

Implements ``HallucinationProbe``, a binary MLP that classifies feature
vectors as truthful (0) or hallucinated (1).  Called from ``solution.py``
via ``evaluate.run_evaluation``.  All four public methods (``fit``,
``fit_hyperparameters``, ``predict``, ``predict_proba``) must be implemented
and their signatures must not change.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class HallucinationProbe:
    """Binary classifier that detects hallucinations from hidden-state features.

    """

    def __init__(self, pca_components: int = 96) -> None:
        self._scaler = StandardScaler()
        self._pca: PCA | None = None
        self._clf: LogisticRegression | None = None
        self._threshold: float = 0.5  # tuned by fit_hyperparameters()
        self._pca_components = max(1, int(pca_components))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train the probe on labelled feature vectors.

        Scales features with ``StandardScaler``, builds the network if needed,
        and optimises with Adam + ``BCEWithLogitsLoss``.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.
            y: Integer label vector of shape ``(n_samples,)``; 0 = truthful,
               1 = hallucinated.

        Returns:
            ``self`` (for method chaining).
        """
        np.random.seed(42)

        X_scaled = self._scaler.fit_transform(X)
        n_components = min(
            self._pca_components,
            X_scaled.shape[1],
            max(16, X_scaled.shape[0] - 1),
        )
        self._pca = PCA(n_components=n_components, svd_solver="auto", random_state=42)
        X_proj = self._pca.fit_transform(X_scaled)

        self._clf = LogisticRegression(
            C=0.03,
            penalty="l2",
            solver="liblinear",
            max_iter=2000,
            random_state=42,
        )
        self._clf.fit(X_proj, y.astype(int))
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune the decision threshold on a validation set to maximise accuracy.

        The chosen threshold is stored in ``self._threshold`` and used by
        subsequent ``predict`` calls.  Call this after ``fit`` and before
        ``predict``.

        Args:
            X_val: Validation feature matrix of shape
                   ``(n_val_samples, feature_dim)``.
            y_val: Integer label vector of shape ``(n_val_samples,)``;
                   0 = truthful, 1 = hallucinated.

        Returns:
            ``self`` (for method chaining).
        """
        probs = self.predict_proba(X_val)[:, 1]

        candidates = np.linspace(0.45, 0.95, 26)

        best_threshold = 0.5
        best_acc = -1.0
        for t in candidates:
            y_pred_t = (probs >= t).astype(int)
            score = accuracy_score(y_val, y_pred_t)
            if score > best_acc:
                best_acc = score
                best_threshold = float(t)

        self._threshold = best_threshold
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels for feature vectors.

        Uses the decision threshold in ``self._threshold`` (default ``0.5``;
        updated by ``fit_hyperparameters``).

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Integer array of shape ``(n_samples,)`` with values in ``{0, 1}``.
        """
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Array of shape ``(n_samples, 2)`` where column 1 contains the
            estimated probability of the hallucinated class (label 1).
            Used to compute AUROC.
        """
        X_scaled = self._scaler.transform(X)
        if self._pca is None or self._clf is None:
            raise RuntimeError("Probe is not fitted. Call fit() first.")
        X_proj = self._pca.transform(X_scaled)
        prob_pos = self._clf.predict_proba(X_proj)[:, 1]
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)

