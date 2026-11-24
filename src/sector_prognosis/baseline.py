"""Iteration 0: Leakage-free reproduction of the original ResNet-MLP + PCA baseline
and naive historical-mean benchmark on Purged Walk-Forward Dev Folds (2012-2019)."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from . import HORIZON_DAYS, SECTORS
from .data import compute_forward_targets, load_raw_merged_df
from .metrics import evaluate_predictions, summarize_fold_metrics
from .splits import get_dev_walk_forward_folds, split_inner_train_val


class OriginalResidualBlock(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.3):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        nn.init.kaiming_uniform_(self.linear.weight, mode="fan_in", nonlinearity="relu")
        self.linear.bias.data.zero_()
        self.bn = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.projection = nn.Linear(input_size, hidden_size) if input_size != hidden_size else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.dropout(self.bn(self.linear(x))))
        if self.projection is not None:
            residual = self.projection(residual)
        return out + residual


class OriginalNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 256, num_hid_layers: int = 4, targets: int = 6):
        super().__init__()
        layers = []
        for i in range(num_hid_layers):
            layers.append(OriginalResidualBlock(input_size if i == 0 else hidden_size, hidden_size))
        self.layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(hidden_size, targets)
        with torch.no_grad():
            self.output_layer.weight *= 0.1
            self.output_layer.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)


def build_original_concat_features(
    raw_feat_matrix: np.ndarray,
    fit_idx: np.ndarray,
    all_idx: np.ndarray,
    offsets: tuple[int, ...] = (0, 20, 62, 125, 250),
    n_components: int = 21,
) -> np.ndarray:
    """Reproduce original MinMaxScaler + PCA(21) + 5-offset concatenation WITHOUT leakage.

    Scaler and PCA are fit strictly on `raw_feat_matrix[fit_idx]` (and its lookback points <= max(fit_idx)).
    """
    max_fit = int(fit_idx.max())
    # Strictly causal: fit only on rows <= max_fit
    scaler = MinMaxScaler()
    scaler.fit(raw_feat_matrix[: max_fit + 1])
    scaled_all = scaler.transform(raw_feat_matrix)

    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(scaled_all[: max_fit + 1])
    pca_all = pca.transform(scaled_all)

    # Concatenate [t - offset for offset in offsets]
    blocks = [pca_all[all_idx - off] for off in offsets]
    return np.concatenate(blocks, axis=1).astype(np.float32)


def train_original_net(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    hidden_size: int = 256,
    num_hid_layers: int = 4,
    lr: float = 3e-4,
    max_epochs: int = 120,
    batch_size: int = 128,
    seed: int = 42,
) -> OriginalNet:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = OriginalNet(x_tr.shape[1], hidden_size=hidden_size, num_hid_layers=num_hid_layers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = nn.MSELoss()

    xt = torch.from_numpy(x_tr).float()
    yt = torch.from_numpy(y_tr).float()
    xv = torch.from_numpy(x_val).float()
    yv = torch.from_numpy(y_val).float()

    best_loss = float("inf")
    best_state = None
    patience = 20
    bad = 0

    n = len(xt)
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            if len(idx) < 2:
                continue
            optimizer.zero_grad()
            loss = criterion(model(xt[idx]), yt[idx])
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            vloss = criterion(model(xv), yv).item()
        if vloss < best_loss - 1e-6:
            best_loss = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def run_iteration_0():
    df = load_raw_merged_df("data")
    y_df, realized = compute_forward_targets(df, horizon=HORIZON_DAYS)
    y_all = y_df.to_numpy(dtype=np.float32)

    # Raw feature matrix matching original notebook: all columns except Date and the 6 prices, plus Month
    feat_cols = [c for c in df.columns if c not in ("Date", *SECTORS)]
    raw_df = df[feat_cols].copy()
    raw_df["Month"] = (df["Date"].dt.month / 12.0).astype(np.float32)
    raw_mat = raw_df.to_numpy(dtype=np.float32)

    valid_mask = (~y_df.isna().any(axis=1)).to_numpy(copy=True)
    valid_mask[:250] = False  # need 250d lookback for offsets=(0..250)

    folds = get_dev_walk_forward_folds(df["Date"], realized, valid_mask)

    results = {"NaiveMean": [], "OriginalResNetPCA": [], "RawRidge": []}

    for fold in folds:
        tr, ev = fold.train_idx, fold.eval_idx
        y_tr_mean = y_all[tr].mean(axis=0)

        # 1. Naive Historical Mean
        pred_naive = np.tile(y_tr_mean, (len(ev), 1))
        m_naive = evaluate_predictions(y_all[ev], pred_naive, y_train_mean=y_tr_mean)
        m_naive["fold"] = fold.name
        results["NaiveMean"].append(m_naive)

        # 2. Original ResNet + PCA (leakage-free!)
        itr, ival = split_inner_train_val(tr, df["Date"], realized)
        X_itr = build_original_concat_features(raw_mat, itr, itr)
        X_ival = build_original_concat_features(raw_mat, itr, ival)
        X_ev = build_original_concat_features(raw_mat, itr, ev)
        net = train_original_net(X_itr, y_all[itr], X_ival, y_all[ival])
        with torch.no_grad():
            pred_net = net(torch.from_numpy(X_ev).float()).numpy()
        m_net = evaluate_predictions(y_all[ev], pred_net, y_train_mean=y_tr_mean)
        m_net["fold"] = fold.name
        results["OriginalResNetPCA"].append(m_net)

        # 3. Ridge on Robust-scaled raw features at decision date t
        scaler = RobustScaler()
        X_tr_r = np.clip(scaler.fit_transform(raw_mat[tr]), -5, 5)
        X_ev_r = np.clip(scaler.transform(raw_mat[ev]), -5, 5)
        ridge = Ridge(alpha=1000.0)
        ridge.fit(X_tr_r, y_all[tr])
        pred_ridge = ridge.predict(X_ev_r)
        m_ridge = evaluate_predictions(y_all[ev], pred_ridge, y_train_mean=y_tr_mean)
        m_ridge["fold"] = fold.name
        results["RawRidge"].append(m_ridge)

    key_cols = [
        "fold", "mean_r2", "oos_r2_vs_train_mean", "rel_r2",
        "rmse", "dir_acc", "mean_ts_ic", "mean_cs_ic",
        "top2_excess_1y_ret", "top2_vs_bot2_spread", "top2_win_rate_vs_ew",
    ]
    for name, rows in results.items():
        print("=" * 95)
        print(f"MODEL: {name}")
        summary = summarize_fold_metrics(rows)
        print(summary[key_cols].to_string(index=False, float_format=lambda x: f"{x:7.4f}"))


if __name__ == "__main__":
    run_iteration_0()
