import copy
import math
import random

import numpy as np
from tqdm import tqdm

import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .config import DEVICE
from .hyperparamters import (
    GRAD_CLIP,
    L1,
    LAMBDA,
)
from .loss import (
    HuberLoss,
    propensity_L_1_loss_objective,
    propensity_L_2_loss_objective,
)
from .model import (
    WeightPredictorNN,
)


def train_representations_learner(
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
        optimizer, mode="min", factor=factor, patience=patience
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
            T_batch = torch.abs((1 - mask_batch).float())

            optimizer.zero_grad()
            z, x_reconstructed, propensity_score, mean, logvar = model(X_batch)
            loss = model.loss_function(
                X_batch, x_reconstructed, z, propensity_score, T_batch, mean, logvar
            )
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
                T_batch = torch.abs((1 - mask_batch).float())
                z, x_reconstructed, propensity_score, mean, logvar = model(X_batch)
                loss = model.loss_function(
                    X_batch, x_reconstructed, z, propensity_score, T_batch, mean, logvar
                )
                epoch_val_loss += loss.item()
        history["val"].append(epoch_val_loss / len(val_loader))

        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_model = model
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        scheduler.step(epoch_val_loss)

    return best_model, history


def train_representations_learner_real_data(
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
        optimizer,
        mode="min",
        factor=factor,
        patience=patience,
    )
    early_stopping_counter = 0

    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        epoch_train_loss = 0.0
        model.train()
        for (
            X_batch,
            _,
            _,
            mask_batch,
        ) in train_loader:
            (
                X_batch,
                mask_batch,
            ) = (
                X_batch.to(device),
                mask_batch.to(device),
            )
            T_batch = torch.abs((1 - mask_batch).float())

            optimizer.zero_grad()
            z, x_reconstructed, propensity_score, mean, logvar = model(X_batch)
            loss = model.loss_function(
                X_batch, x_reconstructed, z, propensity_score, T_batch, mean, logvar
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            epoch_train_loss += loss.item()
        history["train"].append(epoch_train_loss / len(train_loader))

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for (
                X_batch,
                _,
                _,
                mask_batch,
            ) in val_loader:
                X_batch, mask_batch = (
                    X_batch.to(device),
                    mask_batch.to(device),
                )
                T_batch = torch.abs((1 - mask_batch).float())
                z, x_reconstructed, propensity_score, mean, logvar = model(X_batch)
                loss = model.loss_function(
                    X_batch, x_reconstructed, z, propensity_score, T_batch, mean, logvar
                )
                epoch_val_loss += loss.item()
        history["val"].append(epoch_val_loss / len(val_loader))

        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_model = model
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        scheduler.step(epoch_val_loss)

    return best_model, history


def train_weight_predictor(
    treated_y,
    control_y,
    treated_latent,
    control_latent,
    propensity_score_treated,
    propensity_score_control,
    embedding_dim,
    lambda_=LAMBDA,
    L_1=L1,
    epochs=5000,
    exp_num=1,
    batch_size=264,
    verbose=False,
    lr=1e-3,
    return_loss_history=False,
    return_model=False,
    return_weights=True,
):
    seed = 42 + exp_num

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model_2 = WeightPredictorNN(embedding_dim).to(DEVICE)
    optimizer = optim.Adam(model_2.parameters(), lr=lr)
    loss_fn = HuberLoss(delta=1)

    best_loss = float("inf")
    best_model = None
    loss_history = []

    n_treated = treated_latent.shape[0]
    results = {}

    with tqdm(
        total=epochs,
        desc=f"Training Experiment {exp_num}",
        disable=not verbose,
    ) as pbar:
        for epoch in range(epochs):
            model_2.train()
            g = torch.Generator(device="cpu")
            g.manual_seed(seed + epoch)
            perm = torch.randperm(n_treated, generator=g)

            indices = torch.randperm(n_treated, generator=g)[:batch_size]
            # indices = torch.randperm(n_treated)[:batch_size]
            treated_latent_sub = treated_latent[indices]
            treated_y_sub = treated_y[indices]
            propensity_score_treated_sub = propensity_score_treated[indices]

            weights = model_2(treated_latent_sub, control_latent)
            reconstruction = torch.matmul(weights, control_latent)

            reconstruction_loss = loss_fn(reconstruction, treated_latent_sub)

            propensity_L1 = propensity_L_1_loss_objective(
                weights,
                propensity_score_treated_sub,
                propensity_score_control,
                treated_y_sub,
                control_y,
            )

            propensity_L2 = propensity_L_2_loss_objective(
                weights,
                propensity_score_treated_sub,
                propensity_score_control,
                treated_y_sub,
                control_y,
            )

            loss = lambda_ * reconstruction_loss + (1 - lambda_) * (
                L_1 * propensity_L1 + (1 - L_1) * propensity_L2
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_2.parameters(), GRAD_CLIP)
            optimizer.step()

            pbar.update(1)

            cur_loss = loss.item()
            loss_history.append(cur_loss)
            if not math.isnan(cur_loss) and cur_loss < best_loss:
                best_loss = cur_loss
                best_model = copy.deepcopy(model_2)  # Safe for joblib

            if (epoch + 1) % 500 == 0:
                pbar.set_postfix(loss=cur_loss)

    if best_model is None:
        raise RuntimeError(
            "best_model was never set — loss was NaN or invalid throughout training."
        )
    if return_model:
        results["model"] = best_model
    if return_loss_history:
        results["loss_history"] = loss_history
    if return_weights:
        best_model.eval()
        weights = batched_forward_pass(
            best_model, treated_latent, control_latent, batch_size=batch_size
        ).detach()

        results["weights"] = weights
    return results


def batched_forward_pass(model, treated_latent, control_latent, batch_size=64):
    model.eval()
    n_treated = treated_latent.size(0)
    weights = []

    with torch.no_grad():
        for i in range(0, n_treated, batch_size):
            batch_treated = treated_latent[i : i + batch_size]
            batch_weights = model(batch_treated, control_latent)
            weights.append(batch_weights)

    return torch.cat(weights, dim=0)
