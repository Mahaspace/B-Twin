import os
import sys
import time

import numpy as np

import torch

sys.path.append(os.path.abspath("../"))
from dataclasses import dataclass

import numpy.random

from .src import SyncTwin
from .src.config import D_TYPE, DEVICE
from .src.SyncTwin_data_preperation import prepare_real_data
from .src.util import eval_utils, io_utils, train_utils

RANDOM_SEED = 42


@dataclass
class SyncTwinResult:
    model: SyncTwin.SyncTwin
    encoder: torch.nn.Module
    decoder: torch.nn.Module
    decoder_Y: torch.nn.Module | None
    model_path: str
    train_step: int
    step: int
    stats: dict


def run_synctwin(data, data_params, model_params):
    seed = int(model_params["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- unpack params ---
    itr = int(model_params["itr"])
    itr_fine_tune = int(model_params["itr_fine_tune"])
    itr_pretrain = int(model_params["itr_pretrain"])
    lam_recon = float(model_params["lam_recon"])
    lam_prognostic = float(model_params["lam_prognostic"])
    n_hidden = int(model_params["n_hidden"])
    tau = float(model_params["tau"])
    batch_size = int(model_params["batch_size"])

    pretrain_Y = model_params["pretrain_Y"] == "True"
    reduced_fine_tune = model_params["reduced_fine_tune"] == "True"
    linear_decoder = model_params["linear_decoder"] == "True"
    regular = model_params["regular"] == "True"

    base_path_model = "./mimic_data/models"
    base_path_data = "./mimic_data/data"

    train_utils.create_paths(base_path_model, base_path_data)

    # --- prepare training data ---
    prepare_real_data(
        data, data_params["treatment_time"], base_path_data, DEVICE, D_TYPE
    )

    n_control, n_treated, _ = io_utils.load_config(
        base_path_data + "/{}-{}.{}", "train"
    )

    (x_full, t_full, mask_full, batch_ind, y_full, y_control, y_mask, _, _, _) = (
        io_utils.load_real_tensor(base_path_data + "/{}-{}.{}", "train")
    )

    input_dim = x_full.shape[-1]
    train_step = x_full.shape[0]
    step = train_step + y_full.shape[0]

    best_error = float("inf")
    best_model = None
    best_path = None
    training_times = []

    # --- multiple restarts ---
    for i in range(itr):
        start = time.time()
        model_path = f"{base_path_model}/itr-{i}-{{}}.pth"

        # enc / dec
        if regular:
            enc = SyncTwin.RegularEncoder(input_dim, n_hidden)
            dec = SyncTwin.RegularDecoder(enc.hidden_dim, input_dim, train_step)
        else:
            enc = SyncTwin.GRUDEncoder(input_dim, n_hidden)
            dec = SyncTwin.LSTMTimeDecoder(enc.hidden_dim, input_dim, train_step)

        if pretrain_Y:
            if not linear_decoder:
                dec_Y = SyncTwin.RegularDecoder(
                    hidden_dim=enc.hidden_dim,
                    output_dim=y_full.shape[-1],
                    max_seq_len=step - train_step,
                )
            else:
                dec_Y = SyncTwin.LinearDecoder(
                    hidden_dim=enc.hidden_dim,
                    output_dim=y_full.shape[-1],
                    max_seq_len=step - train_step,
                )
        else:
            dec_Y = None
        nsc = SyncTwin.SyncTwin(
            n_control,
            n_treated,
            lam_recon=lam_recon,
            lam_prognostic=lam_prognostic,
            tau=tau,
            encoder=enc,
            decoder=dec,
            decoder_Y=dec_Y,
        )

        # --- pretrain ---
        train_utils.pre_train_reconstruction_loss(
            nsc,
            x_full,
            t_full,
            mask_full,
            x_full,
            t_full,
            mask_full,
            itr_pretrain,
            model_path,
            batch_size,
        )

        if not reduced_fine_tune:
            train_utils.train_all_losses(
                nsc,
                x_full,
                t_full,
                mask_full,
                batch_ind,
                y_full,
                y_control,
                y_mask,
                itr_pretrain,
            )

        # --- validation ---
        effect_est, y_hat = eval_utils.get_treatment_effect(
            nsc, batch_ind, y_full, y_control
        )
        y_control = y_control.to(y_hat.device)
        err = torch.mean(torch.abs(y_control - y_hat[:, :n_control, :])).item()
        training_times.append(time.time() - start)

        if err < best_error:
            best_error = err
            best_model = nsc
            best_path = model_path

    stats = {
        "best_control_error": best_error,
        "avg_training_time": float(np.mean(training_times)),
    }

    return SyncTwinResult(
        model=best_model,
        encoder=best_model.encoder,
        decoder=best_model.decoder,
        decoder_Y=best_model.decoder_Y,
        model_path=best_path,
        train_step=train_step,
        step=step,
        stats=stats,
    )


def load_nsc_(
    nsc,
    x_full,
    t_full,
    mask_full,
    batch_ind_full,
):
    with torch.no_grad():
        C = nsc.get_representation(x_full, t_full, mask_full)
        nsc.update_C0(C, batch_ind_full)
    nsc.eval()


def predict_att_synctwin(
    trained: SyncTwinResult, new_data, data_params, re_train=False
):
    base_path_data = "./mimic_data/data"

    from .src.SyncTwin_data_preperation import prepare_real_data_test

    prepare_real_data_test(
        new_data,
        data_params["treatment_time"],
        base_path_data,
        DEVICE,
        D_TYPE,
    )

    (x, t, mask, batch_ind, y, y_control, y_mask, _, _, _) = io_utils.load_real_tensor(
        base_path_data + "/{}-{}.{}", "test"
    )

    n_control, n_treated, _ = io_utils.load_config(base_path_data + "/{}-{}.{}", "test")

    nsc = SyncTwin.SyncTwin(
        n_control,
        n_treated,
        lam_recon=trained.model.lam_recon,
        lam_prognostic=trained.model.lam_prognostic,
        tau=trained.model.tau,
        encoder=trained.encoder,
        decoder=trained.decoder,
        decoder_Y=trained.decoder_Y,
    )

    load_nsc_(nsc, x, t, mask, batch_ind)
    if re_train:
        train_utils.train_B_self_expressive(
            nsc,
            x,
            t,
            mask,
            batch_ind,
            niters=2000,
            model_path=None,
        )

    effect_est, y_hat = eval_utils.get_treatment_effect(nsc, batch_ind, y, y_control)

    return effect_est.mean().cpu().detach()
