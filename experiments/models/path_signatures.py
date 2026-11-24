"""
Simple Prototype: Rough-Path Depth-2 Signature & Signed Levy-Area Forecaster
---------------------------------------------------------------------------
From Stochastic Analysis / Rough Path Theory (Terry Lyons):
For a multivariate time series X(t) in R^d over a 1-year lookback window:
  - Level-1 Signature: S^(1)_i  = sum_t dX_i(t)               (net 1-year move)
  - Level-2 Signature: S^(2)_ij = sum_{s < t} dX_i(s) dX_j(t) (iterated integral)
  - Signed Levy Area:  A_ij     = 0.5 * (S^(2)_ij - S^(2)_ji)

Why Levy Area is uniquely suited to macro/sector forecasting:
  A_ij is strictly antisymmetric and captures the *chronological order of events*:
    A_ij > 0  ==>  factor i moved BEFORE factor j
    A_ij < 0  ==>  factor j moved BEFORE factor i
  (e.g., "CPI rose before ISM rolled over" vs "ISM rolled over before CPI fell").

Kept deliberately minimal: compress raw 111-dim trajectories via train-only PCA
to d=14 principal economic paths, compute exact Depth-2 Signatures + Levy Areas,
and fit a closed-form Ridge readout. Zero backprop, <1 second runtime!
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def compute_depth2_signatures(paths: np.ndarray) -> np.ndarray:
    """
    Compute exact discrete Depth-2 Signature + Signed Levy Area features.
    Args:
        paths: (N, T, D)
    Returns:
        feats: (N, D + D + D*(D-1)//2) = [latest_level, level1_change, levy_areas]
    """
    N, T, D = paths.shape
    dX = np.diff(paths, axis=1)  # (N, T-1, D)

    # Latest level & Level-1 signature (net change over window)
    latest = paths[:, -1, :]
    lvl1 = dX.sum(axis=1)  # (N, D)

    # Prefix sum of increments strictly before time t: P[t] = sum_{s < t} dX[s]
    prefix = np.zeros_like(dX)
    prefix[:, 1:, :] = np.cumsum(dX[:, :-1, :], axis=1)

    # Level-2 iterated integral: S2[n, i, j] = sum_t prefix[n, t, i] * dX[n, t, j]
    S2 = np.einsum("nti,ntj->nij", prefix, dX)

    # Signed Levy Area (strictly upper triangle of 0.5 * (S2 - S2^T))
    levy = 0.5 * (S2 - np.swapaxes(S2, 1, 2))
    iu = np.triu_indices(D, k=1)
    levy_flat = levy[:, iu[0], iu[1]]  # (N, D*(D-1)//2)

    return np.concatenate([latest, lvl1, levy_flat], axis=1).astype(np.float32)


class PathSignatureForecaster:
    """Minimal Rough-Path Signature + Levy-Area Ridge Forecaster."""

    def __init__(self, n_components: int = 14, ridge_alpha: float = 1000.0):
        self.n_components = n_components
        self.ridge_alpha = ridge_alpha

    def _pack(self, X_sec: np.ndarray, X_macro: np.ndarray) -> np.ndarray:
        N, T, S, Fs = X_sec.shape
        return np.concatenate([X_sec.reshape(N, T, S * Fs), X_macro], axis=-1)

    def fit(self, X_sec: np.ndarray, X_macro: np.ndarray, Y: np.ndarray):
        raw = self._pack(X_sec, X_macro)  # (N, T, D_raw)
        N, T, D_raw = raw.shape
        # Fit PCA strictly on training trajectories
        self.pca = PCA(n_components=self.n_components, random_state=42)
        self.pca.fit(raw.reshape(N * T, D_raw))

        paths = self.pca.transform(raw.reshape(N * T, D_raw)).reshape(N, T, self.n_components)
        Phi = compute_depth2_signatures(paths)
        self.scaler = StandardScaler().fit(Phi)
        self.readout = Ridge(alpha=self.ridge_alpha).fit(self.scaler.transform(Phi), Y)
        return self

    def predict(self, X_sec: np.ndarray, X_macro: np.ndarray) -> np.ndarray:
        raw = self._pack(X_sec, X_macro)
        N, T, D_raw = raw.shape
        paths = self.pca.transform(raw.reshape(N * T, D_raw)).reshape(N, T, self.n_components)
        Phi = compute_depth2_signatures(paths)
        return self.readout.predict(self.scaler.transform(Phi)).astype(np.float32)
