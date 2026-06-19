import logging
import os
import pickle
import sys
from time import time

from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .neural_baselines_ts import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from simulations.simulation import *


def run_experiment_ate(
    dataset="my_sim",
    ihdp_path=None,
    sim_df=None,
    sim_params=None,
    logs=False,
    t_post=84,
):
    """
    dataset:  "ihdp", or "my_sim"
    """
    ## setup logging
    logging.disable(logging.NOTSET)  # re-enable logging if previously disabled
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)  # remove old handlers

    # --- apply new logging mode ---
    if logs:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            force=True,
        )
    else:
        logging.disable(logging.CRITICAL)
    logging.info("Loading dataset: %s", dataset)

    if dataset == "ihdp":
        X, t, y, mu0, mu1 = load_ihdp_1000_rep(ihdp_path)

    elif dataset == "my_sim":
        assert sim_df is not None, "Pass sim_df=generate_simulation(...)."
        X, t, y, mu0, mu1 = load_my_simulation(
            sim_df, treatment_time=sim_params["treatment_time"]
        )

    else:
        raise ValueError("Unknown dataset")
    (
        X_train,
        X_test,
        t_train,
        t_test,
        y_train,
        y_test,
        mu0_train,
        mu0_test,
        mu1_train,
        mu1_test,
    ) = train_test_split(X, t, y, mu0, mu1, test_size=0.3, random_state=SEED)

    results = []

    # TARNet
    logging.info("\ntraining TARNet...")
    tarnet = TARNet(x_dim=X.shape[1], hidden=(200, 200), t_post=t_post).to(DEVICE)
    t1 = time()
    tarnet = train_tarnet(
        tarnet, X_train, t_train, y_train, iters=5000, batch=256, lr=lr
    )
    t2 = time()
    p_y0, p_y1 = predict_tarnet(tarnet, X_test)
    true_ate, pred_ate, ate_err = ate(mu0_test, mu1_test, p_y0, p_y1)
    pehe_val = pehe(mu0_test, mu1_test, p_y0, p_y1)
    # print("TARNet ATE true/pred/error:", true_ate, pred_ate, ate_err, "PEHE:", pehe_val)
    results.append((true_ate, pred_ate, ate_err, pehe_val, t2 - t1))

    # CFRNet
    logging.info("\ntraining CFRNet (MMD)...")
    cfr = CFRNet(x_dim=X.shape[1], hidden=(200, 200), mmd_sigma=1.0, t_post=t_post).to(
        DEVICE
    )
    t1 = time()
    cfr = train_cfrnet(
        cfr,
        X_train,
        t_train,
        y_train,
        mmd_coef=1.0,
        iters=5000,
        batch=256,
        lr=lr,
        mmd_sigma=1.0,
    )
    t2 = time()
    p_y0, p_y1 = predict_tarnet(cfr, X_test)
    true_ate, pred_ate, ate_err = ate(mu0_test, mu1_test, p_y0, p_y1)
    pehe_val = pehe(mu0_test, mu1_test, p_y0, p_y1)
    # print("CFRNet ATE true/pred/error:", true_ate, pred_ate, ate_err, "PEHE:", pehe_val)
    results.append((true_ate, pred_ate, ate_err, pehe_val, t2 - t1))

    # DragonNet
    logging.info("\ntraining DragonNet (with propensity head)...")
    dnet = DragonNet(x_dim=X.shape[1], hidden=(200, 200), t_post=t_post).to(DEVICE)
    t1 = time()
    dnet = train_dragonnet(
        dnet,
        X_train,
        t_train,
        y_train,
        alpha_prop=1.0,
        tr_lambda=1.0,
        use=True,
        iters=5000,
        batch=256,
        lr=lr,
    )
    t2 = time()
    ## alpha and beta are set to 1 in the paper
    p_y0, p_y1, p_prop = predict_dragonnet(dnet, X_test)
    true_ate, pred_ate, ate_err = ate(mu0_test, mu1_test, p_y0, p_y1)
    pehe_val = pehe(mu0_test, mu1_test, p_y0, p_y1)
    # print(
    #     "DragonNet ATE true/pred/error:", true_ate, pred_ate, ate_err, "PEHE:", pehe_val
    # )
    results.append((true_ate, pred_ate, ate_err, pehe_val, t2 - t1))

    # BCAUS-like (propensity balancing); then use simple IPW plug-in to get outcomes
    logging.info("\ntraining BCAUS propensity model...")
    bnet = BCAUS(x_dim=X.shape[1], hidden=(200, 200), t_post=t_post).to(DEVICE)
    t1 = time()
    bnet = train_bcaus(
        bnet,
        X_train,
        t_train,
        y_train,
        alpha_prop=1,
        alpha_balance=10.0,
        iters=500,
        batch=256,
        lr=lr,
    )
    t2 = time()
    p_y0, p_y1, p_prop = predict_bcaus(bnet, X_test)
    true_ate, pred_ate, ate_err = ate(mu0_test, mu1_test, p_y0, p_y1)
    pehe_val = pehe(mu0_test, mu1_test, p_y0, p_y1)

    # print(
    #     "BCAUS ATE true/pred/error:",
    #     true_ate,
    #     pred_ate,
    #     ate_err,
    #     "PEHE:",
    #     pehe_val,
    # )
    results.append((true_ate, pred_ate, ate_err, pehe_val, t2 - t1))

    return results



def run_experiment_att(
    tarnet,
    cfr,
    dnet,
    bnet,
    dataset="my_sim",
    ihdp_path=None,
    sim_df=None,
    sim_params=None,
    logs=False,
    t_post=84,
):
    """
    Runs one experiment and returns ATT metrics for:
        - TARNet
        - CFRNet
        - DragonNet
        - BCAUS
    """

    ## === Logging setup ===
    logging.disable(logging.NOTSET)
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    if logs:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            force=True,
        )
    else:
        logging.disable(logging.CRITICAL)

    logging.info("Loading dataset: %s", dataset)

    ## === Load data ===
    if dataset == "ihdp":
        X, t, y, mu0, mu1 = load_ihdp_1000_rep(ihdp_path)

    elif dataset == "my_sim":
        assert sim_df is not None, "Pass sim_df=generate_simulation(...)."
        X, t, y, mu0, mu1 = load_my_simulation(
            sim_df, treatment_time=sim_params["treatment_time"]
        )

    else:
        raise ValueError("Unknown dataset")

    results = []

    X_test = X.copy()
    t_test = t.copy()
    mu0_test = mu0.copy()
    mu1_test = mu1.copy()
    # ============================================================
    #  TARNet
    # ============================================================

    p_y0, p_y1 = predict_tarnet(tarnet, X_test)
    true_att, pred_att, att_err = att(mu0_test, mu1_test, p_y0, p_y1, t_test)
    pehe_val = pehe_att(mu0_test, mu1_test, p_y0, p_y1, t_test)

    results.append((true_att, pred_att, att_err, pehe_val))

    # ============================================================
    #  CFRNet
    # ============================================================

    p_y0, p_y1 = predict_tarnet(cfr, X_test)
    true_att, pred_att, att_err = att(mu0_test, mu1_test, p_y0, p_y1, t_test)
    pehe_val = pehe_att(mu0_test, mu1_test, p_y0, p_y1, t_test)

    results.append((true_att, pred_att, att_err, pehe_val))

    # ============================================================
    #  DragonNet
    # ============================================================

    p_y0, p_y1, p_prop = predict_dragonnet(dnet, X_test)
    true_att, pred_att, att_err = att(mu0_test, mu1_test, p_y0, p_y1, t_test)
    pehe_val = pehe_att(mu0_test, mu1_test, p_y0, p_y1, t_test)

    results.append((true_att, pred_att, att_err, pehe_val))

    # ============================================================
    #  BCAUS
    # ============================================================

    p_y0, p_y1, p_prop = predict_bcaus(bnet, X_test)
    true_att, pred_att, att_err = att(mu0_test, mu1_test, p_y0, p_y1, t_test)
    pehe_val = pehe(mu0_test, mu1_test, p_y0, p_y1)

    results.append((true_att, pred_att, att_err, pehe_val))

    return results


def run_many_experiments(
    n_experiments,
    params,
    save_dir="./results",
    filename="simulation",
    target="ate",
    verbose=False,
    n_units=500,
    lr=1e-2,
):
    params = params.copy()
    print("Running experiments...")
    print(f"trainig global models on {params['n_units']} individuals")
    df = generate_simulation(**params)
    t_post = params["n_time"] - params["treatment_time"]
    X, t, y, mu0, mu1 = load_my_simulation(df, treatment_time=params["treatment_time"])

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

    tarnet = TARNet(x_dim=X.shape[1], hidden=(200, 200), t_post=t_post).to(DEVICE)
    t1 = time()
    tarnet = train_tarnet(
        tarnet,
        X_train,
        t_train,
        y_train,
        X_val,
        t_val,
        y_val,
        iters=5000,
        batch=256,
        lr=lr,
    )

    cfr = CFRNet(x_dim=X.shape[1], hidden=(200, 200), mmd_sigma=1.0, t_post=t_post).to(
        DEVICE
    )
    cfr = train_cfrnet(
        cfr,
        X_train,
        t_train,
        y_train,
        X_val,
        t_val,
        y_val,
        mmd_coef=1.0,
        iters=5000,
        batch=256,
        lr=lr,
        mmd_sigma=1.0,
    )
    dnet = DragonNet(x_dim=X.shape[1], hidden=(200, 200), t_post=t_post).to(DEVICE)

    dnet = train_dragonnet(
        dnet,
        X_train,
        t_train,
        y_train,
        X_val,
        t_val,
        y_val,
        alpha_prop=1.0,
        tr_lambda=1.0,
        use_tr=True,
        iters=5000,
        batch=256,
        lr=lr,
    )

    bnet = BCAUS(x_dim=X.shape[1], hidden=(200, 200), t_post=t_post).to(DEVICE)

    bnet = train_bcaus(
        bnet,
        X_train,
        t_train,
        y_train,
        X_val,
        t_val,
        y_val,
        alpha_prop=1,
        alpha_balance=10.0,
        iters=5000,
        batch=256,
        lr=lr,
    )
    params["n_units"] = n_units
    os.makedirs(save_dir, exist_ok=True)
    filename = filename + "_" + target
    if target == "ate":
        run_experiment = run_experiment_ate
    elif target == "att":
        run_experiment = run_experiment_att
    else:
        raise ValueError("Unknown target: choose 'ate' or 'att'")
    results = Parallel(n_jobs=1)(
        delayed(run_experiment)(
            dataset="my_sim",
            sim_params=params,
            logs=False,
            sim_df=generate_simulation(**params),
            t_post=params["n_time"] - params["treatment_time"],
            tarnet=tarnet,
            cfr=cfr,
            dnet=dnet,
            bnet=bnet,
        )
        for _ in tqdm(
            range(n_experiments),
            total=n_experiments,
            desc="Simulations",
            disable=not verbose,
        )
    )
    # final filename
    fname = (
        f"{filename}_bias={params['bias']}"
        f"_const={params['constant_effect']}"
        f"_sigma={params['sigma']}"
        f"_alpha={params['alpha']}.pkl"
    )
    results = np.array(results)
    save_path = os.path.join(save_dir, fname)
    with open(save_path, "wb") as f:
        pickle.dump(results, f)

    print(f"\nSaved {filename} in : {save_path}")
    return results
