import os
import pickle
import sys
import time

import pandas as pd
from joblib import Parallel, delayed

import torch

from .att import att
from .config import (
    DEVICE,
)
from .data_preperation import (
    process_dataset,
    subset,
)
from .train_utils import (
    train_weight_predictor,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from simulations.simulation import generate_simulation


def run_experiment(
    i, params, model, weight_epochs=5000, ground_truth=False, lambda_=0.5, L1=0.7
):
    params["seed"] = i
    print(f"Running iteration {i}, seed : {i}")

    t1 = time.time()

    data = generate_simulation(**params)
    data = data[["id", "time", "y_0", "y", "treatment", "propensity_score"]]
    test_ids = {
        "control": data[data.treatment == 0]["id"].unique(),
        "treated": data[data.treatment == 1]["id"].unique(),
    }
    test = pd.concat(
        [subset(data, test_ids["control"], 0), subset(data, test_ids["treated"], 1)]
    )
    datasets, configs = process_dataset([test], ["test"], params)

    X_r_test = torch.tensor(datasets["test"]["X_r"], dtype=torch.float32).to(DEVICE)
    y_f_test = torch.tensor(datasets["test"]["y_f"], dtype=torch.float32).to(DEVICE)

    latent_representations, _, _ = model.encoder(X_r_test)

    n_c = configs["test"]["n_control_units"]
    n_t = configs["test"]["n_treated_units"]
    n_total = n_c + n_t

    # Mean outcomes
    y_treated = X_r_test[n_c:, :, :].mean(axis=1)
    y_control = X_r_test[:n_c, :, :].mean(axis=1)

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
        exp_num=i,
        lambda_=lambda_,
        L_1=L1,
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
        exp_num=i,
        lambda_=lambda_,
        L_1=L1,
    )
    weights_c = results_c["weights"]
    control_data = y_f_test[:n_c, :, -1].detach()
    treated_data = y_f_test[n_c:, :, -1].detach()

    p0, p1 = n_c / n_total, n_t / n_total
    att_1 = att(weights_t, prop_t, prop_c, treated_data, control_data)
    att_2 = -att(weights_c, prop_c, prop_t, control_data, treated_data)
    effect_est = p1 * att_1 + p0 * att_2

    t2 = time.time()
    return effect_est, (t2 - t1)


def run_att_experiment_old(
    n_experiments,
    params,
    model,
    save_dir="./results",
    filename=None,
    weight_epochs=5000,
    ground_truth=False,
    lambda_=0.5,
    L1=1,
    save=True,
):
    # print("Running experiments...")

    results = Parallel(n_jobs=1)(
        delayed(run_experiment)(
            i, params, model, weight_epochs, ground_truth, lambda_, L1
        )
        for i in range(n_experiments)
    )
    ate_list, inference_times = zip(*results)
    if save:
        os.makedirs(save_dir, exist_ok=True)
        filename += f"att_results_bias={params['bias']}_constant_effect={params['constant_effect']}_sigma={params['sigma']}_alpha={params['alpha']}.pkl"

        save_path = os.path.join(save_dir, filename)
        with open(save_path, "wb") as f:
            pickle.dump(ate_list, f)
            pickle.dump(inference_times, f)
    return ate_list


def run_att_experiment(
    n_experiments,
    params,
    model,
    save_dir="./results",
    filename=None,
    weight_epochs=5000,
    ground_truth=False,
    lambda_=0.7,
    L1=0,
    save=True,
    max_retries_per_exp=20,  # safety guard
):
    ate_list = []
    inference_times = []

    exp_idx = 0  # how many successful experiments we have
    global_seed = 0  # keeps changing seeds even after failures

    while exp_idx < n_experiments:
        retries = 0
        success = False

        while not success and retries < max_retries_per_exp:
            try:
                result = run_experiment(
                    global_seed,
                    params,
                    model,
                    weight_epochs,
                    ground_truth,
                    lambda_,
                    L1,
                )
                ate, t = result

                ate_list.append(ate)
                inference_times.append(t)

                success = True
                exp_idx += 1
                print(f"✓ Experiment {exp_idx}/{n_experiments} succeeded")

            except Exception as e:
                retries += 1
                print(f"✗ Experiment failed (seed={global_seed}, retry={retries}): {e}")

            finally:
                global_seed += 1  # always move seed forward

        if not success:
            raise RuntimeError(
                f"Experiment failed {max_retries_per_exp} times in a row. Aborting."
            )

    if save:
        os.makedirs(save_dir, exist_ok=True)
        filename = (
            filename
            + f"att_results_bias={params['bias']}"
            + f"_constant_effect={params['constant_effect']}"
            + f"_sigma={params['sigma']}"
            + f"_alpha={params['alpha']}.pkl"
        )

        save_path = os.path.join(save_dir, filename)
        with open(save_path, "wb") as f:
            pickle.dump(ate_list, f)
            pickle.dump(inference_times, f)

    return ate_list
