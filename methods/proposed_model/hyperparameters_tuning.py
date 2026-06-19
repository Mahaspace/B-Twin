import itertools
import logging
import os
import pickle
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src.config import DEVICE
from src.data_format import make_loader
from src.data_preperation import prepare_data
from src.model import CausalInferenceModel
from src.train_utils import train_representations_learner

import torch
from torch import optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from simulations.simulation import generate_simulation
from src.data_preperation import process_dataset, subset
from src.train_utils import train_weight_predictor

# =========================
# Logging configuration
# =========================

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


def compute_treatment_effects(df, T_0):
    """
    Computes individual, average, and treated average treatment effects.

    Parameters:
    - df: DataFrame with columns ['Y_0', 'Y_1', 'treatment', 'time', 'treated']
    -T_0: int, time step when treatment starts

    Returns:
    - A dictionary with ITE, ATE, and ATT
    """
    # Compute individual treatment effects

    post_treatment = df[df["time"] >= T_0]
    post_treatment["ITE"] = post_treatment["y_1"] - post_treatment["y_0"]

    # Average Treatment Effect (ATE)
    ATE = post_treatment["ITE"].mean()

    # Average Treatment Effect on the Treated (ATT)
    treated_units = post_treatment[post_treatment["treatment"] == 1]
    ATT = treated_units["ITE"].mean()

    return {"ITE": post_treatment["ITE"], "ATE": ATE, "ATT": ATT}


## generate a biased placebo dataset
# ============================================================
def biased_placebo_dataset(data, params, bias_strength=0.5, seed=42):
    """
    Introduce bias into the placebo dataset by correlating the treatment assignment
    with a confounding variable in the data.
    """
    ## fix random seed for reproducibility
    np.random.seed(seed)
    data = data.copy()
    control_data = data[data.treatment == 0]
    n_times = params["n_time"]
    # Generate a confounding variable from pre_treatment data
    pre_treatment_data = control_data[control_data.time < params["treatment_time"]]
    confounder = pre_treatment_data.groupby("id")["y"].mean()
    confounder = (confounder - confounder.min()) / (confounder.max() - confounder.min())
    # Adjust treatment assignment based on the confounder
    treatment_prob = 0.5 + bias_strength * (confounder - 0.5)
    treatment_assignment = np.random.binomial(1, treatment_prob)

    control_data["treatment"] = np.repeat(treatment_assignment, n_times)
    return control_data


def plot_heatmap(
    objective_values, params, idx1=0, idx2=1, save_path="../../figures/", placebo=False
):
    """
    idx1, idx2 = indexes of hyperparams to visualize.
    For your case: 0=alpha, 1=beta, 2=gamma, 3=lambda, 4=L1
    """

    rows = []
    for hparams, loss in objective_values.items():
        rows.append(list(hparams) + [loss])

    df = pd.DataFrame(rows, columns=["alpha", "beta", "gamma", "lambda", "L1", "loss"])

    pivot = df.pivot_table(
        values="loss",
        index=df.columns[idx1],
        columns=df.columns[idx2],
        aggfunc="mean",
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", annot_kws={"size": 12})
    plt.title(f"Loss Heatmap: {pivot.index.name} vs {pivot.columns.name}")

    # === Save figure ===
    # filename = filename = (
    #     f"heatmap_{pivot.index.name}_vs_{pivot.columns.name}_placebo_{placebo}_simulation_params_bias_{params['bias']}_constant_effect_{params['constant_effect']}_sigma_{params['sigma']}_alpha_{params['alpha']}.jpg"
    # )

    # full_path = os.path.join(save_path, filename)
    # plt.savefig(full_path)
    # print(f"Saved heatmap plot to: {save_path}")
    plt.show()


# ============================================================
# === Estimate ATT for given hyperparameters α, β, γ ==========
# ============================================================
def weights_regression(
    data,
    params,
    model,
    weight_epochs=5000,
    ground_truth=False,
    lambda_=0.5,
    L1=0.7,
    lr=1e-4,
    batch_size=256,
):
    t1 = time.time()

    data = data[["id", "time", "y_0", "y", "treatment", "propensity_score"]]
    test_ids = {
        "control": data[data.treatment == 0]["id"].unique(),
        "treated": data[data.treatment == 1]["id"].unique(),
    }
    test = pd.concat(
        [subset(data, test_ids["control"], 0), subset(data, test_ids["treated"], 1)]
    )
    datasets, configs = process_dataset([test], ["test"], params, counterfactual=True)

    X_r_test = torch.tensor(datasets["test"]["X_r"], dtype=torch.float32).to(DEVICE)
    y_f_test = torch.tensor(datasets["test"]["y_f"], dtype=torch.float32).to(DEVICE)

    latent_representations, _, _ = model.encoder(X_r_test)

    n_c = configs["test"]["n_control_units"]
    n_t = configs["test"]["n_treated_units"]
    n_total = n_c + n_t

    # Mean outcomes
    y_treated = X_r_test[n_c:, :, :].mean(axis=1)
    y_control = X_r_test[:n_c, :, :].mean(axis=1)
    y_cf_test = torch.tensor(
        datasets["test"]["y_counterfactual"], dtype=torch.float32
    ).to(DEVICE)

    y_cf_treated = y_cf_test[n_c:, :, -1]
    y_cf_control = y_cf_test[:n_c, :, -1]

    if ground_truth:
        prop_scores = torch.tensor(
            datasets["test"]["propensity_score"], dtype=torch.float32
        ).to(DEVICE)

    else:
        prop_scores = model.propensity_estimator(latent_representations).squeeze()

    prop_t = prop_scores[n_c:].detach()
    prop_c = prop_scores[:n_c].detach()

    rep_t = latent_representations[n_c:].detach()
    rep_c = latent_representations[:n_c].detach()

    w_t = 1 / prop_t.unsqueeze(1)
    w_c = 1 / prop_c.unsqueeze(1)

    rep_t = torch.cat([rep_t, w_t], dim=1)
    rep_c = torch.cat([rep_c, w_c], dim=1)

    results_t = train_weight_predictor(
        y_treated,
        y_control,
        rep_t,
        rep_c,
        prop_t,
        prop_c,
        embedding_dim=latent_representations.shape[1],
        epochs=weight_epochs,
        lambda_=lambda_,
        L_1=L1,
        lr=lr,
        batch_size=batch_size,
    )
    weights_t = results_t["weights"]
    results_c = train_weight_predictor(
        y_control,
        y_treated,
        rep_c,
        rep_t,
        prop_c,
        prop_t,
        embedding_dim=latent_representations.shape[1],
        epochs=weight_epochs,
        lambda_=lambda_,
        L_1=L1,
        lr=lr,
        batch_size=batch_size,
    )
    weights_c = results_c["weights"]
    control_y = y_f_test[:n_c, :, -1].detach()
    treated_y = y_f_test[n_c:, :, -1].detach()

    ## estimate ite
    pred_y_0_t = weights_t @ control_y
    pred_y_1_c = weights_c @ treated_y

    ite_est_t = treated_y - pred_y_0_t
    true_ite_t = treated_y - y_cf_treated
    ite_est_c = pred_y_1_c - control_y
    true_ite_c = y_cf_control - control_y
    ## concatenate
    ite_est = torch.cat([ite_est_c, ite_est_t], dim=0)
    true_ite = torch.cat([true_ite_c, true_ite_t], dim=0)

    ite_error = torch.mean(torch.abs(ite_est - true_ite) ** 2)
    # return float
    return ite_error.cpu().detach().item()


def compute_error(
    alpha,
    beta,
    gamma,
    lambda_,
    L1,
    data,
    simulation_params,
    batch_size=256,
    num_epochs=100,
    weight_epochs=1000,
    lr_1=1e-3,
    lr_2=1e-4,
):
    datasets, configs = prepare_data(data, simulation_params)
    train_loader = make_loader(datasets, configs, "train", batch_size)
    val_loader = make_loader(datasets, configs, "val", batch_size)

    # Initialize representation model
    model = CausalInferenceModel(
        n_features=1,
        seq_len=84,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=lr_1)

    best_model, history = train_representations_learner(
        model, train_loader, val_loader, optimizer, num_epochs=num_epochs, device=DEVICE
    )
    model = best_model
    ite_error = weights_regression(
        data,
        model=model,
        params=simulation_params,
        weight_epochs=weight_epochs,
        ground_truth=True,
        lambda_=lambda_,
        L1=L1,
        lr=lr_2,
        batch_size=batch_size,
    )

    return ite_error


# ============================================================
# === Hyperparameter tuning + Sensitivity Analysis Plot ======
# ============================================================


def tune(
    simulation_params,
    save_path="../../figures",
    random_search=False,
    n_random_samples=100,
    max_workers=4,
    placebo=False,
):
    ## extensive grid search over hyperparameters
    # Hyperparameter lists
    alpha_list = [1]
    beta_list = [0.005, 0.01, 0.1, 0.5, 1]
    # beta_list = [0.005]
    gamma_list = [0.01, 0.1, 0.5, 1]
    lambda_list = [0.1, 0.2, 0.5, 0.7, 0.9]
    # L1_list = [0, 0.3, 0.5, 0.7, 1.0]
    L1_list = [1.0]
    # Training params
    batch_size = 256
    num_epochs = 100
    weight_epochs = 1000
    lr1 = 1e-3
    lr2 = 1e-4

    # Generate combinations
    if random_search:
        all_combinations = [
            (
                random.choice(alpha_list),
                random.choice(beta_list),
                random.choice(gamma_list),
                random.choice(lambda_list),
                random.choice(L1_list),
            )
            for _ in range(n_random_samples)
        ]
    else:
        all_combinations = list(
            itertools.product(alpha_list, beta_list, gamma_list, lambda_list, L1_list)
        )

    objective_values = {}
    best_ite_error = float("inf")
    best_hyperparams = None
    data = generate_simulation(**simulation_params)
    if placebo:
        biased_data = biased_placebo_dataset(data, simulation_params)
        data = biased_data.copy()

    # Function to evaluate a single combination
    def evaluate_combo(params):
        alpha, beta, gamma, lambda_, L1 = params
        ite_error = compute_error(
            alpha,
            beta,
            gamma,
            lambda_,
            L1,
            data,
            simulation_params,
            batch_size,
            num_epochs,
            weight_epochs,
            lr1,
            lr2,
        )
        return params, ite_error

    # Parallel execution
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_params = {
            executor.submit(evaluate_combo, params): params
            for params in all_combinations
        }
        for future in as_completed(future_to_params):
            params, ite_error = future.result()
            objective_values[params] = ite_error

            if ite_error < best_ite_error:
                best_ite_error = ite_error
    #             best_hyperparams = params

    # # # Visualization
    # plot_heatmap(
    #     objective_values, simulation_params, idx1=1, idx2=2, placebo=placebo
    # )  # beta vs gamma

    # plot_heatmap(
    #     objective_values, simulation_params, idx1=3, idx2=2, placebo=placebo
    # )  # lambda vs gamma
    # Save objective_values in pkl file
    with open(
        f"./results/hyperparameter_tuning_objective_values_placebo_{placebo}_simulation_params_bias_{simulation_params['bias']}_constant_effect_{simulation_params['constant_effect']}_sigma_{simulation_params['sigma']}_alpha_{simulation_params['alpha']}.pkl",
        "wb",
    ) as f:
        pickle.dump(objective_values, f)


if __name__ == "__main__":
    params = [
        {
            "n_units": 1000,
            "n_time": 84 * 2,
            "treatment_time": 84,
            "phi": 0.8,
            "sigma": 5,
            "alpha": 0.05,
            "seed": 42,
            "bias": True,
            "constant_effect": True,
        },
        {
            "n_units": 1000,
            "n_time": 84 * 2,
            "treatment_time": 84,
            "phi": 0.8,
            "sigma": 5,
            "alpha": 0.05,
            "seed": 42,
            "bias": True,
            "constant_effect": False,
        },
    ]

    n_features = 1
    seq_len = 84
    logger.info("Starting main execution")
    for param in params:
        logger.info(f"Running placebo=True | params={param}")
        tune(
            param,
            max_workers=1,
            random_search=False,
            placebo=True,
        )
        logger.info(f"Running placebo=False | params={param}")
        tune(
            param,
            max_workers=1,
            random_search=False,
            placebo=False,
        )
        logger.info("Finished all experiments")
