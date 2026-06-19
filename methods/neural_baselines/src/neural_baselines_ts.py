# causal_baselines.py
"""
Runable script with implementations of:
- TARNet
- CFRNet (MMD penalty)
- DragonNet (+ optional Targeted Regularization)
- BCAUS-like propensity balancing network

Produces simple synthetic confounded data, dfs each model, and prints
ATE and PEHE comparisons.

Dependencies: see requirements.txt
"""

import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Metrics
# ---------------------------
def ate(true_y0, true_y1, pred_y0, pred_y1):
    # average treatment effect (ground truth minus predicted)
    true_ate = float(np.mean(true_y1 - true_y0))
    pred_ate = float(np.mean(pred_y1 - pred_y0))
    return true_ate, pred_ate, abs(pred_ate - true_ate)


def pehe(true_y0, true_y1, pred_y0, pred_y1):
    # root mean squared error of individual treatment effects
    true_tau = true_y1 - true_y0
    pred_tau = pred_y1 - pred_y0
    return float(np.sqrt(np.mean((true_tau - pred_tau) ** 2)))


def att(mu0, mu1, y0_hat, y1_hat, t_test):
    mask = t_test == 1
    true_att = (mu1[mask] - mu0[mask]).mean()
    pred_att = (y1_hat[mask] - y0_hat[mask]).mean()
    att_err = abs(true_att - pred_att)
    return true_att, pred_att, att_err


def pehe_att(mu0, mu1, y0_hat, y1_hat, t_test):
    mask = t_test == 1
    return np.sqrt(
        np.mean(((mu1[mask] - mu0[mask]) - (y1_hat[mask] - y0_hat[mask])) ** 2)
    )


# ---------------------------
# Neural building blocks
# ---------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims=(200, 200), activation=nn.ReLU):
        super().__init__()
        dims = [in_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------
# TARNet
# ---------------------------
class TARNet(nn.Module):
    def __init__(self, x_dim, t_post, rep_dim=200, hidden=(200, 200)):
        super().__init__()
        self.repr = MLP(x_dim, hidden_dims=hidden)
        self.head0 = nn.Sequential(
            nn.Linear(hidden[-1], hidden[-1]), nn.ReLU(), nn.Linear(hidden[-1], t_post)
        )
        self.head1 = nn.Sequential(
            nn.Linear(hidden[-1], hidden[-1]), nn.ReLU(), nn.Linear(hidden[-1], t_post)
        )

    def forward(self, x):
        r = self.repr(x)
        y0 = self.head0(r).squeeze(-1)
        y1 = self.head1(r).squeeze(-1)
        return y0, y1, r


# ---------------------------
# CFRNet (adds MMD between treated and control reps)
# ---------------------------
def gaussian_kernel_matrix(x, y, sigma=1.0):
    # x: (n_x, d), y: (n_y, d)
    x_norm = (x**2).sum(1).view(-1, 1)
    y_norm = (y**2).sum(1).view(1, -1)
    K = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
    return torch.exp(-K / (2 * sigma**2))


def mmd_rbf(x, y, sigma=1.0):
    Kxx = gaussian_kernel_matrix(x, x, sigma)
    Kyy = gaussian_kernel_matrix(y, y, sigma)
    Kxy = gaussian_kernel_matrix(x, y, sigma)
    m = x.size(0)
    n = y.size(0)
    return Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()


class CFRNet(TARNet):
    def __init__(self, x_dim, t_post, rep_dim=200, hidden=(200, 200), mmd_sigma=1.0):
        super().__init__(x_dim, t_post, rep_dim, hidden)
        self.mmd_sigma = mmd_sigma

    # forward same as TARNet; mmd computed in dfing loop


# ---------------------------
# DragonNet
# ---------------------------
class DragonNet(nn.Module):
    def __init__(self, x_dim, t_post, hidden=(200, 200)):
        super().__init__()
        self.repr = MLP(x_dim, hidden_dims=hidden)
        rdim = hidden[-1]
        # outcome heads
        self.head0 = nn.Sequential(
            nn.Linear(rdim, rdim), nn.ReLU(), nn.Linear(rdim, t_post)
        )
        self.head1 = nn.Sequential(
            nn.Linear(rdim, rdim), nn.ReLU(), nn.Linear(rdim, t_post)
        )
        # propensity head (logit)
        self.propensity = nn.Sequential(
            nn.Linear(rdim, rdim // 2), nn.ReLU(), nn.Linear(rdim // 2, 1)
        )

    def forward(self, x):
        r = self.repr(x)
        y0 = self.head0(r).squeeze(-1)
        y1 = self.head1(r).squeeze(-1)
        p = torch.sigmoid(self.propensity(r)).squeeze(-1)
        return y0, y1, p, r


# ---------------------------
# BCAUS ( same as dragonnet but with an auto balancing loss)
# ---------------------------
class BCAUS(DragonNet):
    def __init__(self, x_dim, t_post, hidden=(200, 200)):
        super().__init__(x_dim=x_dim, t_post=t_post, hidden=hidden)


def evaluate_validation(model, X_val, t_val, y_val, loss_fn):
    model.eval()
    with torch.no_grad():
        y0, y1, *rest = model(X_val)
        q = t_val.unsqueeze(1) * y1 + (1 - t_val).unsqueeze(1) * y0
        val_loss = loss_fn(q, y_val).item()
    model.train()
    return val_loss


# ======================================================================
# TRAINERS WITH VALIDATION + EARLY STOPPING
# ======================================================================


def train_tarnet(
    model,
    X_df,
    t_df,
    y_df,
    X_val=None,
    t_val=None,
    y_val=None,
    iters=2000,
    batch=256,
    lr=1e-3,
    patience=20,
    eval_freq=50,
):
    model = model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X = torch.from_numpy(X_df).float().to(DEVICE)
    t = torch.from_numpy(t_df).float().to(DEVICE)
    y = torch.from_numpy(y_df).float().to(DEVICE)

    if X_val is not None:
        X_val = torch.from_numpy(X_val).float().to(DEVICE)
        t_val = torch.from_numpy(t_val).float().to(DEVICE)
        y_val = torch.from_numpy(y_val).float().to(DEVICE)

    n = X.shape[0]
    best_loss = np.inf
    patience_left = patience
    best_state = None

    for epoch in range(iters):
        rep = np.random.choice(n, batch, replace=False)
        xb, tb, yb = X[rep], t[rep], y[rep]

        y0, y1, _ = model(xb)
        q = tb.unsqueeze(1) * y1 + (1 - tb).unsqueeze(1) * y0
        loss = loss_fn(q, yb)

        opt.zero_grad()
        loss.backward()
        opt.step()

        # ---- Validation --------------------------------------------------
        if (epoch % eval_freq == 0) and (X_val is not None):
            val_loss = evaluate_validation(model, X_val, t_val, y_val, loss_fn)

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_left = patience
            else:
                patience_left -= 1
                if patience_left == 0:
                    break

    # restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# ======================================================================
# CFRNet WITH VALIDATION
# ======================================================================
def train_cfrnet(
    model,
    X_df,
    t_df,
    y_df,
    X_val=None,
    t_val=None,
    y_val=None,
    mmd_coef=1.0,
    iters=2000,
    batch=256,
    lr=1e-3,
    mmd_sigma=1.0,
    patience=20,
    eval_freq=50,
):
    model = model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X = torch.from_numpy(X_df).float().to(DEVICE)
    t = torch.from_numpy(t_df).float().to(DEVICE)
    y = torch.from_numpy(y_df).float().to(DEVICE)

    if X_val is not None:
        X_val = torch.from_numpy(X_val).float().to(DEVICE)
        t_val = torch.from_numpy(t_val).float().to(DEVICE)
        y_val = torch.from_numpy(y_val).float().to(DEVICE)

    n = X.shape[0]
    best_loss = np.inf
    patience_left = patience
    best_state = None

    for epoch in range(iters):
        rep = np.random.choice(n, batch, replace=False)
        xb, tb, yb = X[rep], t[rep], y[rep]

        y0, y1, repz = model(xb)

        q = tb.unsqueeze(1) * y1 + (1 - tb).unsqueeze(1) * y0
        pred_loss = loss_fn(q, yb)

        rep_t = repz[tb == 1]
        rep_c = repz[tb == 0]
        if rep_t.shape[0] > 1 and rep_c.shape[0] > 1:
            mmd = mmd_rbf(rep_t, rep_c, sigma=mmd_sigma)
        else:
            mmd = torch.tensor(0.0, device=DEVICE)

        loss = pred_loss + mmd_coef * mmd

        opt.zero_grad()
        loss.backward()
        opt.step()

        # ---- Validation ---------------------------------------------------
        if (epoch % eval_freq == 0) and (X_val is not None):
            val_loss = evaluate_validation(model, X_val, t_val, y_val, loss_fn)

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_left = patience
            else:
                patience_left -= 1
                if patience_left == 0:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# ======================================================================
# DRAGONNET WITH VALIDATION
# ======================================================================
def train_dragonnet(
    model,
    X_df,
    t_df,
    y_df,
    X_val=None,
    t_val=None,
    y_val=None,
    alpha_prop=1.0,
    tr_lambda=1.0,
    use_tr=False,
    iters=2000,
    batch=256,
    lr=1e-3,
    patience=20,
    eval_freq=50,
):
    model = model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    bce = nn.BCELoss()

    X = torch.from_numpy(X_df).float().to(DEVICE)
    t = torch.from_numpy(t_df).float().to(DEVICE)
    y = torch.from_numpy(y_df).float().to(DEVICE)

    if X_val is not None:
        X_val = torch.from_numpy(X_val).float().to(DEVICE)
        t_val = torch.from_numpy(t_val).float().to(DEVICE)
        y_val = torch.from_numpy(y_val).float().to(DEVICE)

    n = X.shape[0]
    best_loss = np.inf
    patience_left = patience
    best_state = None

    for epoch in range(iters):
        rep = np.random.choice(n, batch, replace=False)
        xb, tb, yb = X[rep], t[rep], y[rep]

        y0, y1, p, repz = model(xb)

        q = tb.unsqueeze(1) * y1 + (1 - tb).unsqueeze(1) * y0

        loss_y = mse(q, yb)
        loss_prop = bce(p, tb)

        tr_loss = 0.0
        if use_tr:
            g = torch.clamp(p, 1e-3, 1 - 1e-3)
            clever = tb / g - (1 - tb) / (1 - g)
            resid = (yb - q).detach()
            tr_loss = torch.mean(clever.unsqueeze(1) * resid) ** 2

        loss = loss_y + alpha_prop * loss_prop + tr_lambda * tr_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        # ---- Validation ---------------------------------------------------
        if (epoch % eval_freq == 0) and (X_val is not None):
            val_loss = evaluate_validation(model, X_val, t_val, y_val, mse)

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_left = patience
            else:
                patience_left -= 1
                if patience_left == 0:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# ======================================================================
# BCAUSS WITH VALIDATION
# ======================================================================
def train_bcaus(
    model,
    X_df,
    t_df,
    y_df,
    X_val=None,
    t_val=None,
    y_val=None,
    alpha_prop=1.0,
    alpha_balance=10.0,
    iters=2000,
    batch=256,
    lr=1e-3,
    patience=20,
    eval_freq=50,
):
    model = model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCELoss()
    mse = nn.MSELoss()

    X = torch.from_numpy(X_df).float().to(DEVICE)
    t = torch.from_numpy(t_df).float().to(DEVICE)
    y = torch.from_numpy(y_df).float().to(DEVICE)

    if X_val is not None:
        X_val = torch.from_numpy(X_val).float().to(DEVICE)
        t_val = torch.from_numpy(t_val).float().to(DEVICE)
        y_val = torch.from_numpy(y_val).float().to(DEVICE)

    n = X.shape[0]
    best_loss = np.inf
    patience_left = patience
    best_state = None

    for epoch in range(iters):
        rep = np.random.choice(n, batch, replace=False)
        xb, tb, yb = X[rep], t[rep], y[rep]

        y0, y1, p, repz = model(xb)

        q = tb.unsqueeze(1) * y1 + (1 - tb).unsqueeze(1) * y0
        loss_y = mse(q, yb)

        loss_prop = bce(p, tb)

        g = torch.clamp(p, 1e-3, 1 - 1e-3)
        w_t = tb / g
        w_c = (1 - tb) / (1 - g)

        w_tn = w_t / (w_t.sum() + 1e-9)
        w_cn = w_c / (w_c.sum() + 1e-9)

        mean_t = (w_tn.unsqueeze(1) * xb).sum(0)
        mean_c = (w_cn.unsqueeze(1) * xb).sum(0)

        imbalance = torch.sum((mean_t - mean_c) ** 2)

        loss = loss_y + alpha_prop * loss_prop + alpha_balance * imbalance

        opt.zero_grad()
        loss.backward()
        opt.step()

        # ---- Validation ---------------------------------------------------
        if (epoch % eval_freq == 0) and (X_val is not None):
            val_loss = evaluate_validation(model, X_val, t_val, y_val, mse)

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_left = patience
            else:
                patience_left -= 1
                if patience_left == 0:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# ---------------------------
# Predict helpers
# ---------------------------
def predict_tarnet(model, X):
    model = model.to(DEVICE).eval()
    with torch.no_grad():
        Xb = torch.from_numpy(X).to(DEVICE)
        y0, y1, _ = model(Xb)
        return y0.cpu().numpy(), y1.cpu().numpy()


def predict_dragonnet(model, X):
    model = model.to(DEVICE).eval()
    with torch.no_grad():
        Xb = torch.from_numpy(X).to(DEVICE)
        y0, y1, p, _ = model(Xb)
        return y0.cpu().numpy(), y1.cpu().numpy(), p.cpu().numpy()


def predict_bcaus(model, X):
    model = model.to(DEVICE).eval()
    with torch.no_grad():
        Xb = torch.from_numpy(X).to(DEVICE)
        y0, y1, p, _ = model(Xb)
        return y0.cpu().numpy(), y1.cpu().numpy(), p.cpu().numpy()


def load_ihdp_1000_rep(data_path, rep=0):
    df = np.load(data_path)
    X = df.f.x.copy()
    T = df.f.t.copy()
    YF = df.f.yf.copy()
    YCF = df.f.ycf.copy()
    mu_0 = df.f.mu0.copy()
    mu_1 = df.f.mu1.copy()

    t, y, x, mu0, mu_1 = T[:, rep], YF[:, rep], X[:, :, rep], mu_0[:, rep], mu_1[:, rep]

    return t, y, x, mu0, mu_1


def load_my_simulation(df, treatment_time=84):
    """
    Convert simulation DataFrame into causal ML format (unit-level).

    Returns:
        X      : shape (N, T_pre)    pre-treatment outcomes
        t      : shape (N,)          treatment assignment (0/1)
        y      : shape (N,)          factual post-treatment average outcome
        mu0    : shape (N,)          counterfactual mean outcome under control
        mu1    : shape (N,)          counterfactual mean outcome under treatment
    """

    # Sort to make reshape safe
    df = df.sort_values(["id", "time"])

    n_steps = df.time.nunique()
    n_units = int(df.id.count() / n_steps)

    # Extract arrays
    y = df["y"].values.reshape(n_units, n_steps)
    y0 = df["y_0"].values.reshape(n_units, n_steps)
    y1 = df["y_1"].values.reshape(n_units, n_steps)
    t = df["treatment"].values.reshape(n_units, n_steps)[:, 0]  # constant across time
    X = y[:, :treatment_time]  # shape (N, T_pre)
    y_factual = y[:, treatment_time:]
    # ----------------------------
    # True μ0 and μ1 (post-treatment means)
    # ----------------------------
    mu0 = y0[:, treatment_time:]
    mu1 = y1[:, treatment_time:]

    return (
        X.astype(np.float32),
        t.astype(np.float32),
        y_factual.astype(np.float32),
        mu0.astype(np.float32),
        mu1.astype(np.float32),
    )


def load_my_data(df, treatment_time=84):
    """
    Convert data into causal ML format (unit-level). There is no effect here so potential outcomes equal observed outcomes

    Returns:
        X      : shape (N, T_pre)    pre-treatment outcomes
        t      : shape (N,)          treatment assignment (0/1)
        y      : shape (N,)          factual post-treatment average outcome
        mu0    : shape (N,)          counterfactual mean outcome under control
        mu1    : shape (N,)          counterfactual mean outcome under treatment
    """

    # Sort to make reshape safe
    df = df.sort_values(["id", "time"])

    n_steps = df.time.nunique()
    n_units = int(df.id.count() / n_steps)
    # Extract arrays
    y = df["y"].values.reshape(n_units, n_steps)
    y0 = df["y"].values.reshape(n_units, n_steps)
    y1 = df["y"].values.reshape(n_units, n_steps)
    t = df["treatment"].values.reshape(n_units, n_steps)[:, 0]  # constant across time
    X = y[:, :treatment_time]  # shape (N, T_pre)
    y_factual = y[:, treatment_time:]
    # ----------------------------
    # True μ0 and μ1 (post-treatment means)
    # ----------------------------
    mu0 = y0[:, treatment_time:]
    mu1 = y1[:, treatment_time:]

    return (
        X.astype(np.float32),
        t.astype(np.float32),
        y_factual.astype(np.float32),
        mu0.astype(np.float32),
        mu1.astype(np.float32),
    )
