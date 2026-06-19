import os
import sys
import time

import numpy as np
import numpy.random

import torch
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from dataclasses import dataclass

from methods.simulations.simulation import generate_simulation
from methods.Synctwin.src import SyncTwin

# from methods.Synctwin.src import SyncTwin
from methods.Synctwin.src.config import (
    D_TYPE,
    DEVICE,
    params_exp,
)
from methods.Synctwin.src.SyncTwin_data_preperation import (
    prepare_data,
)
from methods.Synctwin.src.util import (
    eval_utils,
    io_utils,
    train_utils,
)


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


def build_synctwin_modules(
    input_dim,
    y_dim,
    train_step,
    step,
    n_hidden,
    regular,
    pretrain_Y,
    linear_decoder,
):
    if regular:
        enc = SyncTwin.RegularEncoder(input_dim, n_hidden)
        dec = SyncTwin.RegularDecoder(enc.hidden_dim, input_dim, train_step)
    else:
        enc = SyncTwin.GRUDEncoder(input_dim, n_hidden)
        dec = SyncTwin.LSTMTimeDecoder(enc.hidden_dim, input_dim, train_step)

    dec_Y = None
    if pretrain_Y:
        DecoderY = SyncTwin.LinearDecoder if linear_decoder else SyncTwin.RegularDecoder
        dec_Y = DecoderY(enc.hidden_dim, y_dim, step - train_step)

    return enc, dec, dec_Y


def run_synctwin(
    simulation_cfg,
    params_exp=params_exp,
    model_dir="./models",
    data_dir="./data",
):
    seed = int(params_exp["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    itr = int(params_exp["itr"])
    itr_fine_tune = int(params_exp["itr_fine_tune"])
    itr_pretrain = int(params_exp["itr_pretrain"])
    lam_recon = float(params_exp["lam_recon"])
    lam_prognostic = float(params_exp["lam_prognostic"])
    n_hidden = int(params_exp["n_hidden"])
    tau = float(params_exp["tau"])
    batch_size = int(params_exp["batch_size"])

    pretrain_Y = params_exp["pretrain_Y"] == "True"
    reduced_fine_tune = params_exp["reduced_fine_tune"] == "True"
    linear_decoder = params_exp["linear_decoder"] == "True"
    regular = params_exp["regular"] == "True"

    train_utils.create_paths(model_dir, data_dir)
    data_path = data_dir + "/{}-{}.{}"

    # --- generate re data ---
    data = generate_simulation(**simulation_cfg)
    prepare_data(data, simulation_cfg["treatment_time"], data_dir, DEVICE, D_TYPE)

    # --- load training + validation ---
    n_c_tr, n_t_tr, _ = io_utils.load_config(data_path, "train")
    n_c_val, n_t_val, _ = io_utils.load_config(data_path, "val")

    (x_tr, t_tr, m_tr, b_tr, y_tr, y_c_tr, y_m_tr, _, _, _, _) = io_utils.load_tensor(
        data_path, "train"
    )

    (x_val, t_val, m_val, b_val, y_val, y_c_val, y_m_val, _, _, _, _) = (
        io_utils.load_tensor(data_path, "val")
    )

    input_dim = x_tr.shape[-1]
    y_dim = y_tr.shape[-1]
    train_step = x_tr.shape[0]
    step = train_step + y_tr.shape[0]

    best_error = float("inf")
    best_model = None
    best_model_path = None
    training_times = []

    # --- restarts ---
    for i in range(itr):
        start = time.time()
        model_path = f"{model_dir}/itr-{i}-{{}}.pth"

        enc, dec, dec_Y = build_synctwin_modules(
            input_dim,
            y_dim,
            train_step,
            step,
            n_hidden,
            regular,
            pretrain_Y,
            linear_decoder,
        )

        nsc = SyncTwin.SyncTwin(
            n_c_tr,
            n_t_tr,
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
            x_tr,
            t_tr,
            m_tr,
            x_val,
            t_val,
            m_val,
            itr_pretrain,
            model_path,
            batch_size,
        )

        if not reduced_fine_tune:
            train_utils.train_all_losses(
                nsc, x_tr, t_tr, m_tr, b_tr, y_tr, y_c_tr, y_m_tr, itr_pretrain
            )

        # --- validation fine-tune ---
        nsc_val = SyncTwin.SyncTwin(
            n_c_val,
            n_t_val,
            lam_recon=lam_recon,
            lam_prognostic=lam_prognostic,
            tau=tau,
            encoder=enc,
            decoder=dec,
            decoder_Y=dec_Y,
        )

        train_utils.load_pre_train_and_init(
            nsc_val, x_val, t_val, m_val, b_val, model_path, init_decoder_Y=pretrain_Y
        )

        train_utils.train_B_self_expressive(
            nsc_val, x_val, t_val, m_val, b_val, itr_fine_tune, model_path
        )
        effect_est, y_hat = eval_utils.get_treatment_effect(
            nsc_val, b_val, y_val, y_c_val
        )
        y_control = y_c_val.to(y_hat.device)
        err = torch.mean(torch.abs(y_control - y_hat[:, :n_c_val, :])).item()

        training_times.append(time.time() - start)

        if err < best_error:
            best_error = err
            best_model = nsc_val
            best_model_path = model_path
    stats = {
        "avg_training_time": float(np.mean(training_times)),
    }
    return SyncTwinResult(
        model=best_model,
        encoder=best_model.encoder,
        decoder=best_model.decoder,
        decoder_Y=best_model.decoder_Y,
        model_path=best_model_path,
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


def train_B_self_expressive(
    nsc,
    x_full,
    t_full,
    mask_full,
    batch_ind_full,
    niters=20000,
    model_path="models/sync/{}.pth",
    batch_size=None,
    lr=1e-3,
    test_freq=1,
):
    # mini-batch training not implemented
    assert batch_size is None

    optimizer = optim.Adam([nsc.B], lr=lr)

    best_loss = 10000
    best_model = None

    with torch.no_grad():
        C = nsc.get_representation(x_full, t_full, mask_full)

    for itr in range(1, niters + 1):
        optimizer.zero_grad()

        B_reduced = nsc.get_B_reduced(batch_ind_full)

        loss = nsc.self_expressive_loss(C, B_reduced)

        loss.backward()
        optimizer.step()

        if itr % test_freq == 0:
            with torch.no_grad():
                print("Iter {:04d} | Total Loss {:.6f}".format(itr, loss.item()))
                if np.isnan(loss.item()):
                    return 1
                if loss < best_loss:
                    best_loss = loss
                    best_model = nsc
    return best_model


def predict_att_synctwin(
    trained: SyncTwinResult,
    new_data,
    data_params,
    base_path_data="./methods/Synctwin/data",
    re_train=False,
    n_iter_re_train=2000,
):
    from methods.Synctwin.src.SyncTwin_data_preperation import prepare_data_test

    prepare_data_test(
        new_data,
        data_params["treatment_time"],
        base_path_data,
        DEVICE,
        D_TYPE,
    )

    (x, t, mask, batch_ind, y, y_control, y_mask, _, _, _, _) = io_utils.load_tensor(
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
        nsc = train_B_self_expressive(
            nsc,
            x,
            t,
            mask,
            batch_ind,
            niters=n_iter_re_train,
            model_path=None,
        )

    effect_est, _ = eval_utils.get_treatment_effect(nsc, batch_ind, y, y_control)

    return effect_est.mean().cpu().detach()


if __name__ == "__main__":
    params = {
        "seed": "100",
        "model_id": "",
        "itr": "1",
        "itr_pretrain": "50",
        "itr_fine_tune": "20",
        "batch_size": "100",
        "pretrain_Y": "True",
        "reduced_fine_tune": "True",
        "linear_decoder": "False",
        "lam_prognostic": "1",
        "lam_recon": "1",
        "tau": "1",
        "n_hidden": "20",
        "sim_id": None,
        "regular": "True",
    }

    simulation_params = {
        "n_units": 100,
        "n_time": 168,
        "treatment_time": 84,
        "phi": 0.8,
        "sigma": 5,
        "alpha": 0.05,
        "seed": None,
        "bias": True,
        "constant_effect": True,
    }

    synctwin = run_synctwin(params_exp=params, simulation_cfg=simulation_params)
    att_est = predict_att_synctwin(
        trained=synctwin,
        new_data=generate_simulation(**simulation_params),
        data_params=simulation_params,
    )
