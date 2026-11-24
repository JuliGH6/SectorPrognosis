"""
Paradigm 6: Multi-Scale Liquid Reservoir Computing (Echo State Network)
-----------------------------------------------------------------------
Financial macro datasets have ~15-25 non-overlapping years of history, so large
gradient-trained networks easily overfit.

A Multi-Scale Leaky Echo State Reservoir bypasses backprop overfitting completely:
1. Feeds the 16-step macro+sector trajectory through a *fixed* sparse random
   recurrent dynamical system (spectral radius rho=0.92) with heterogeneous
   leak rates alpha_i in [0.1, 0.9] (creating fast & slow dynamical memory).
2. Extracts rich nonlinear fading-memory states (final reservoir state +
   temporal mean + temporal std).
3. Solves the multi-output readout analytically in closed form via Ridge Regression!
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge


class LiquidReservoirForecaster:
    """Multi-scale Leaky Echo State Network with closed-form Ridge readout."""

    def __init__(
        self,
        reservoir_size: int = 384,
        spectral_radius: float = 0.92,
        sparsity: float = 0.90,
        input_scale: float = 0.35,
        ridge_alpha: float = 500.0,
        seed: int = 42,
    ):
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.input_scale = input_scale
        self.ridge_alpha = ridge_alpha
        self.seed = seed
        self._initialized = False

    def _init_weights(self, in_dim: int):
        rng = np.random.default_rng(self.seed)
        H = self.reservoir_size
        self.W_in = rng.uniform(-1.0, 1.0, size=(in_dim, H)).astype(np.float32) * self.input_scale

        # Sparse recurrent reservoir matrix scaled to exact spectral radius
        W = rng.normal(0.0, 1.0, size=(H, H)).astype(np.float32)
        mask = rng.random((H, H)) < self.sparsity
        W[mask] = 0.0
        eigvals = np.linalg.eigvals(W)
        max_eig = np.max(np.abs(eigvals)) + 1e-8
        self.W_res = (W / max_eig) * self.spectral_radius

        # Multi-timescale leak rates: slow (0.1) to fast (0.9)
        self.alpha = np.linspace(0.1, 0.9, H, dtype=np.float32)[None, :]
        self._initialized = True

    def _transform(self, X_sec: np.ndarray, X_macro: np.ndarray) -> np.ndarray:
        N, T, S, Fs = X_sec.shape
        X = np.concatenate([X_sec.reshape(N, T, S * Fs), X_macro], axis=-1)
        if not self._initialized:
            self._init_weights(X.shape[-1])

        H = self.reservoir_size
        state = np.zeros((N, H), dtype=np.float32)
        states = np.zeros((N, T, H), dtype=np.float32)

        for t in range(T):
            pre = X[:, t, :] @ self.W_in + state @ self.W_res
            cand = np.tanh(pre)
            state = (1.0 - self.alpha) * state + self.alpha * cand
            states[:, t, :] = state

        # Readout features: final state + mean state + temporal volatility of reservoir
        feat = np.concatenate(
            [states[:, -1, :], states.mean(axis=1), states.std(axis=1)], axis=1
        )
        return feat

    def fit(self, X_sec: np.ndarray, X_macro: np.ndarray, Y: np.ndarray):
        Phi = self._transform(X_sec, X_macro)
        self.readout = Ridge(alpha=self.ridge_alpha)
        self.readout.fit(Phi, Y)
        return self

    def predict(self, X_sec: np.ndarray, X_macro: np.ndarray) -> np.ndarray:
        Phi = self._transform(X_sec, X_macro)
        return self.readout.predict(Phi).astype(np.float32)
