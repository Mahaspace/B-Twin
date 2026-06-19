import random
import time

import numpy as np
import pandas as pd

import torch
from torch import optim

from .src.att import att
from .src.config import DEVICE
from .src.data_format import make_loader_from_real_data
from .src.data_preperation import (
    prepare_real_data,
    process_real_dataset,
    subset,
)
from .src.model import CausalInferenceModel
from .src.train_utils import (
    batched_forward_pass,
    train_representations_learner_real_data,
    train_weight_predictor,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # # Make CUDA deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_btwin(data, data_params, model_params, seed=42):
    set_seed(seed)
    seq_len = data_params["treatment_time"]
    n_features = data_params["n_features"]
    batch_size = model_params["batch_size"]
    train_epochs = model_params["train_epochs"]
    weight_epochs = model_params["weight_epochs"]
    lr1 = model_params["lr1"]
    lambda_ = model_params["lambda_"]
    L1 = model_params["L1"]
    EMBEDDING_DIM = model_params["embedding_dim"]
    HIDDEN_SIZE = model_params["hidden_dim"]
    BETA = model_params["beta"]
    GAMMA = model_params["gamma"]
    ALPHA = model_params["alpha"]
    PROPENSITY_HIDDEN_DIMS = model_params["propensity_hidden_dims"]
    add_test_split = model_params["add_test_split"]
    EPSILON = model_params["epsilon"]
    datasets, configs = prepare_real_data(
        data, data_params, add_test_split=add_test_split
    )

    train_loader = make_loader_from_real_data(
        datasets, configs, "train", batch_size=64 * 4
    )
    val_loader = make_loader_from_real_data(datasets, configs, "val", batch_size=64 * 4)
    representation_model = CausalInferenceModel(
        n_features=n_features,
        seq_len=seq_len,
        emb_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_SIZE,
        beta=BETA,
        gamma=GAMMA,
        alpha=ALPHA,
        propensity_hidden_dims=PROPENSITY_HIDDEN_DIMS,
        epsilon=EPSILON,
    )

    representation_model.to(DEVICE)

    optimizer = optim.Adam(representation_model.parameters(), lr=lr1)

    representation_model, history = train_representations_learner_real_data(
        representation_model,
        train_loader,
        val_loader,
        optimizer,
        train_epochs,
        device=DEVICE,
    )

    torch.cuda.empty_cache()

    test_ids = {
        "control": data[data.treatment == 0]["id"].unique(),
        "treated": data[data.treatment == 1]["id"].unique(),
    }
    test = pd.concat(
        [subset(data, test_ids["control"], 0), subset(data, test_ids["treated"], 1)]
    )
    datasets_test, configs_test = process_real_dataset([test], ["test"], data_params)

    X_r_test = datasets_test["test"]["X_r"].clone().detach().to(DEVICE)
    y_f_test = datasets_test["test"]["y_f"].clone().detach().to(DEVICE)
    representation_model.eval()
    with torch.no_grad():
        latent_representations, _, _ = representation_model.encoder(X_r_test)

    n_c = configs_test["test"]["n_control_units"]
    n_t = configs_test["test"]["n_treated_units"]
    n_total = n_c + n_t

    # Mean outcomes
    y_treated = X_r_test[n_c:, :, :].mean(axis=1)
    y_control = X_r_test[:n_c, :, :].mean(axis=1)

    prop_scores = representation_model.propensity_estimator(
        latent_representations
    ).squeeze()

    prop_t = prop_scores[n_c:].detach()
    prop_c = prop_scores[:n_c].detach()

    rep_t = latent_representations[n_c:].detach()
    rep_c = latent_representations[:n_c].detach()

    w_t = 1 / prop_t.unsqueeze(1)
    w_c = 1 / prop_c.unsqueeze(1)

    rep_t = torch.cat([rep_t, w_t], dim=1)
    rep_c = torch.cat([rep_c, w_c], dim=1)
    t1 = time.time()
    results_t = train_weight_predictor(
        y_treated,
        y_control,
        rep_t,
        rep_c,
        prop_t,
        prop_c,
        embedding_dim=latent_representations.shape[1],
        epochs=weight_epochs,
        exp_num=1,
        lambda_=lambda_,
        L_1=L1,
        batch_size=batch_size,
        lr=1e-4,
        return_loss_history=False,
        return_model=True,
        return_weights=False,
    )
    weight_predictor_t = results_t["model"]

    results_c = train_weight_predictor(
        y_control,
        y_treated,
        rep_c,
        rep_t,
        prop_c,
        prop_t,
        embedding_dim=latent_representations.shape[1],
        epochs=weight_epochs,
        exp_num=0,
        lambda_=lambda_,
        L_1=L1,
        batch_size=batch_size,
        lr=1e-4,
        return_loss_history=False,
        return_model=True,
        return_weights=False,
    )
    weight_predictor_c = results_c["model"]

    return representation_model, weight_predictor_t, weight_predictor_c


def predict_att_btwin(data, data_params, models):
    representation_model, weight_predictor_t, weight_predictor_c = models
    seq_len = data_params["treatment_time"]
    n_features = data_params["n_features"]

    torch.cuda.empty_cache()

    test_ids = {
        "control": data[data.treatment == 0]["id"].unique(),
        "treated": data[data.treatment == 1]["id"].unique(),
    }
    test = pd.concat(
        [subset(data, test_ids["control"], 0), subset(data, test_ids["treated"], 1)]
    )
    datasets_test, configs_test = process_real_dataset([test], ["test"], data_params)

    representation_model.to(DEVICE)
    weight_predictor_t.to(DEVICE)
    weight_predictor_c.to(DEVICE)

    representation_model.eval()
    weight_predictor_t.eval()
    weight_predictor_c.eval()

    with torch.no_grad():
        X_r_test = datasets_test["test"]["X_r"].clone().detach().to(DEVICE)
        y_f_test = datasets_test["test"]["y_f"].clone().detach().to(DEVICE)

        z, x_reconstructed, propensity_score, mean, logvar = representation_model(
            X_r_test
        )

        n_c = configs_test["test"]["n_control_units"]
        n_t = configs_test["test"]["n_treated_units"]
        n_total = n_c + n_t
        control_data = y_f_test[:n_c, :, -1].detach()
        treated_data = y_f_test[n_c:, :, -1].detach()
        prop_scores = representation_model.propensity_estimator(z).squeeze()
        prop_t = prop_scores[n_c:].detach()
        prop_c = prop_scores[:n_c].detach()

        rep_t = z[n_c:].detach()
        rep_c = z[:n_c].detach()

        w_t = 1 / prop_t.unsqueeze(1)
        w_c = 1 / prop_c.unsqueeze(1)

        rep_t = torch.cat([rep_t, w_t], dim=1)
        rep_c = torch.cat([rep_c, w_c], dim=1)

        p0, p1 = n_c / n_total, n_t / n_total
        weights_t = batched_forward_pass(
            weight_predictor_t, rep_t, rep_c, batch_size=256
        )
        weights_c = batched_forward_pass(
            weight_predictor_c, rep_c, rep_t, batch_size=256
        )

        att_1 = att(weights_t, prop_t, prop_c, treated_data, control_data)
        att_2 = -att(weights_c, prop_c, prop_t, control_data, treated_data)
        effect_est = p1 * att_1 + p0 * att_2

        return effect_est.cpu().numpy()
