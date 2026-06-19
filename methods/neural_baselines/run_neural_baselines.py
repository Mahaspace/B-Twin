import time

from sklearn.model_selection import train_test_split

from .src.neural_baselines_ts import *


def run_tarnet(data, data_params, model_params=None):
    df = data.copy()
    t_post = data_params["n_time"] - data_params["treatment_time"]
    X, t, y, mu0, mu1 = load_my_data(df, treatment_time=data_params["treatment_time"])

    (
        X_train,
        X_val,
        t_train,
        t_val,
        y_train,
        y_val,
        mu0_train,
        mu0_val,
        mu1_train,
        mu1_val,
    ) = train_test_split(X, t, y, mu0, mu1, test_size=0.2, random_state=SEED)

    tnet = TARNet(x_dim=X.shape[1], hidden=(200, 200), t_post=t_post).to(DEVICE)
    t1 = time.time()
    tnet = train_tarnet(
        tnet,
        X_train,
        t_train,
        y_train,
        X_val,
        t_val,
        y_val,
        iters=5000,
        batch=256,
        lr=1e-3,
    )

    return tnet


def predict_att_tarnet(model, data, data_params):
    df = data.copy()
    X, t, y, _, _ = load_my_data(df, treatment_time=data_params["treatment_time"])
    p_y0, p_y1 = predict_tarnet(model, X)
    mask = t == 1
    pred_att = (p_y1[mask] - p_y0[mask]).mean()
    # pred_att = (y[mask] - p_y0[mask]).mean()
    return pred_att


def run_cfrnet(data, data_params, model_params=None):
    df = data.copy()
    t_post = data_params["n_time"] - data_params["treatment_time"]
    X, t, y, mu0, mu1 = load_my_data(df, treatment_time=data_params["treatment_time"])

    (
        X_train,
        X_val,
        t_train,
        t_val,
        y_train,
        y_val,
        mu0_train,
        mu0_val,
        mu1_train,
        mu1_val,
    ) = train_test_split(X, t, y, mu0, mu1, test_size=0.2, random_state=SEED)

    cfr = CFRNet(x_dim=X.shape[1], hidden=(200, 200), t_post=t_post).to(DEVICE)
    t1 = time.time()
    cfr = train_cfrnet(
        cfr,
        X_train,
        t_train,
        y_train,
        X_val,
        t_val,
        y_val,
        iters=5000,
        batch=256,
        lr=1e-3,
    )

    return cfr


def predict_att_cfrnet(model, data, data_params):
    df = data.copy()
    X, t, y, _, _ = load_my_data(df, treatment_time=data_params["treatment_time"])
    p_y0, p_y1 = predict_tarnet(model, X)
    mask = t == 1
    pred_att = (p_y1[mask] - p_y0[mask]).mean()
    # pred_att = (y[mask] - p_y0[mask]).mean()
    return pred_att


def run_dragonnet(data, data_params, model_params=None):
    df = data.copy()
    t_post = data_params["n_time"] - data_params["treatment_time"]
    X, t, y, mu0, mu1 = load_my_data(df, treatment_time=data_params["treatment_time"])

    (
        X_train,
        X_val,
        t_train,
        t_val,
        y_train,
        y_val,
        mu0_train,
        mu0_val,
        mu1_train,
        mu1_val,
    ) = train_test_split(X, t, y, mu0, mu1, test_size=0.2, random_state=SEED)

    dnet = DragonNet(x_dim=X.shape[1], hidden=(200, 200), t_post=t_post).to(DEVICE)
    t1 = time.time()
    dnet = train_dragonnet(
        dnet,
        X_train,
        t_train,
        y_train,
        X_val,
        t_val,
        y_val,
        iters=5000,
        batch=256,
        lr=1e-3,
    )

    return dnet


def predict_att_dragonnet(model, data, data_params):
    df = data.copy()
    X, t, y, _, _ = load_my_data(df, treatment_time=data_params["treatment_time"])
    p_y0, p_y1, p_prop = predict_dragonnet(model, X)
    mask = t == 1
    pred_att = (p_y1[mask] - p_y0[mask]).mean()
    # pred_att = (y[mask] - p_y0[mask]).mean()
    return pred_att


def run_bcauss(data, data_params, model_params=None):
    df = data.copy()
    t_post = data_params["n_time"] - data_params["treatment_time"]
    X, t, y, mu0, mu1 = load_my_data(df, treatment_time=data_params["treatment_time"])

    (
        X_train,
        X_val,
        t_train,
        t_val,
        y_train,
        y_val,
        mu0_train,
        mu0_val,
        mu1_train,
        mu1_val,
    ) = train_test_split(X, t, y, mu0, mu1, test_size=0.2, random_state=SEED)

    bnet = BCAUS(x_dim=X.shape[1], hidden=(200, 200), t_post=t_post).to(DEVICE)
    t1 = time.time()
    bnet = train_bcaus(
        bnet,
        X_train,
        t_train,
        y_train,
        X_val,
        t_val,
        y_val,
        iters=5000,
        batch=256,
        lr=1e-3,
    )

    return bnet


def predict_att_bcauss(model, data, data_params):
    df = data.copy()
    X, t, y, _, _ = load_my_data(df, treatment_time=data_params["treatment_time"])
    p_y0, p_y1, p_prop = predict_bcaus(model, X)
    mask = t == 1
    pred_att = (p_y1[mask] - p_y0[mask]).mean()
    # pred_att = (y[mask] - p_y0[mask]).mean()
    return pred_att
