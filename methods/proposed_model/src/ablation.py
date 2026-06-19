import os
import pickle
import sys
import time

import pandas as pd
import torch
from joblib import Parallel, delayed
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .config import (
    DEVICE,
)
from .data_preperation import (
    process_dataset,
    subset,
)
from .hyperparamters import GRAD_CLIP
from .loss import (
    HuberLoss,
)
from .model import NormalWeightPredictorNN

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from simulations.simulation import generate_simulation


def train_representations_learner_C1(
    model,
    train_loader,
    val_loader,
    optimizer,
    num_epochs=10,
    device="cpu",
    patience=5,
    factor=0.5,
):
    history = {"train": [], "val": []}
    best_model = None
    best_loss = float("inf")
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=factor, patience=patience, verbose=True
    )
    early_stopping_counter = 0

    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        epoch_train_loss = 0.0
        model.train()
        for X_batch, _, _, mask_batch, p_batch in train_loader:
            X_batch, mask_batch, p_batch = (
                X_batch.to(device),
                mask_batch.to(device),
                p_batch.to(device),
            )

            optimizer.zero_grad()
            z, x_reconstructed, mean, logvar = model(X_batch)
            loss = model.loss_function(X_batch, x_reconstructed, mean, logvar)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            epoch_train_loss += loss.item()
        history["train"].append(epoch_train_loss / len(train_loader))

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for X_batch, _, _, mask_batch, p_batch in val_loader:
                X_batch, mask_batch, p_batch = (
                    X_batch.to(device),
                    mask_batch.to(device),
                    p_batch.to(device),
                )

                z, x_reconstructed, mean, logvar = model(X_batch)
                loss = model.loss_function(X_batch, x_reconstructed, mean, logvar)
                epoch_val_loss += loss.item()
        history["val"].append(epoch_val_loss / len(val_loader))

        # print(
        #     f"Epoch {epoch+1}/{num_epochs}, Train Loss: {history['train'][-1]:.4f}, Val Loss: {history['val'][-1]:.4f}"
        # )

        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_model = model
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        scheduler.step(epoch_val_loss)

    return best_model, history


def train_weight_predictor(
    treated_latent,
    control_latent,
    embedding_dim,
    epochs=5000,
    exp_num=1,
    batch_size=64,
    verbose=False,
):
    model_2 = NormalWeightPredictorNN(embedding_dim).to(DEVICE)
    optimizer = optim.Adam(model_2.parameters(), lr=1e-4)
    loss_fn = HuberLoss(delta=1.0)  # Use Huber Loss instead of MSE

    best_loss = float("inf")
    best_model = None
    n_treated = treated_latent.shape[0]
    with tqdm(
        total=epochs,
        desc=f"Training Experiment {exp_num}",
        position=0,
        leave=True,
        disable=not verbose,
    ) as pbar:
        for epoch in range(epochs):
            model_2.train()
            indices = torch.randperm(n_treated)[:batch_size]
            treated_latent_sub = treated_latent[indices]
            weights = model_2(treated_latent_sub, control_latent)
            reconstruction = torch.matmul(weights, control_latent)
            reconstruction_loss = loss_fn(reconstruction, treated_latent_sub)

            loss = reconstruction_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_2.parameters(), GRAD_CLIP)
            optimizer.step()

            pbar.update(1)  # Update progress bar

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_model = model_2

            if (epoch + 1) % 500 == 0:  # Print loss every 500 epochs
                pbar.set_postfix(loss=loss.item())
    best_model.eval()
    return batched_forward_pass(
        best_model, treated_latent, control_latent, batch_size=batch_size
    ).detach()


def batched_forward_pass(model, treated_latent, control_latent, batch_size=128):
    model.eval()
    n_treated = treated_latent.size(0)
    weights = []

    with torch.no_grad():
        for i in range(0, n_treated, batch_size):
            batch_treated = treated_latent[i : i + batch_size]
            batch_weights = model(batch_treated, control_latent)
            weights.append(batch_weights)

    return torch.cat(weights, dim=0)


def run_experiment(i, params, model, weight_epochs=5000, ground_truth=False):
    # print(f"Running iteration {i}")
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

    # Mean outcomes
    y_treated = X_r_test[n_c:, :, :].mean(axis=1)
    y_control = X_r_test[:n_c, :, :].mean(axis=1)

    rep_t = latent_representations[n_c:].detach()
    rep_c = latent_representations[:n_c].detach()

    weights = train_weight_predictor(
        rep_t,
        rep_c,
        embedding_dim=latent_representations.shape[1],
        epochs=weight_epochs,
        exp_num=i,
    )

    y_treated = y_f_test[n_c:, :, -1]
    y_control = y_f_test[:n_c, :, -1]
    effect_est = (y_treated - (weights @ y_control)).mean()

    t2 = time.time()

    return effect_est, (t2 - t1)


def run_att_experiment_old(
    n_experiments,
    params,
    model,
    save_dir="./results_att",
    filename=None,
    weight_epochs=5000,
    ground_truth=False,
):
    # print("Running experiments...")
    os.makedirs(save_dir, exist_ok=True)

    results = Parallel(n_jobs=6)(
        delayed(run_experiment)(i, params, model, weight_epochs, ground_truth)
        for i in range(n_experiments)
    )

    ate_list, inference_times = zip(*results)

    filename += f"att_results_bias={params['bias']}_constant_effect={params['constant_effect']}_sigma={params['sigma']}_alpha={params['alpha']}.pkl"

    save_path = os.path.join(save_dir, filename)
    with open(save_path, "wb") as f:
        pickle.dump(ate_list, f)
        pickle.dump(inference_times, f)

    print(f"Results {filename} saved to {save_path}")


def run_att_experiment(
    n_experiments,
    params,
    model,
    save_dir="./results",
    filename=None,
    weight_epochs=5000,
    ground_truth=False,
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
