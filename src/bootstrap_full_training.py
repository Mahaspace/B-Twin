import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import logging
import time

from methods.neural_baselines.run_neural_baselines import (
    predict_att_bcauss,
    predict_att_cfrnet,
    predict_att_dragonnet,
    predict_att_tarnet,
    run_bcauss,
    run_cfrnet,
    run_dragonnet,
    run_tarnet,
)
from methods.proposed_model.run_b_twin import predict_att_btwin, run_btwin
from methods.Synctwin.run_synctwin import predict_att_synctwin, run_synctwin


def setup_logger(name="bootstrap", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def select_bootsrap_treated_samples(data, n_samples, seed=None):
    rng = np.random.default_rng(seed)
    data.set_index("id", drop=False, inplace=True)

    ids = data.index.unique()
    sampled_ids = rng.choice(ids, size=n_samples, replace=True)

    bootstrap_data = data.loc[sampled_ids].reset_index(drop=True)

    return bootstrap_data


def run_att_bootstrap(
    data,
    n_bootsraps,
    data_params,
    b_twin_model_params,
    synctwin_model_params,
    logger=None,
):
    if logger is None:
        logger = setup_logger("ATT-Bootstrap")

    logger.info("Starting ATT bootstrap")
    logger.info(f"Number of bootstrap samples: {n_bootsraps}")
    logger.info(f"Dataset size: {len(data)}")

    att_estimates = {
        "b_twin": [],
        "tnet": [],
        "cfr": [],
        "dnet": [],
        "bnet": [],
        "synctwin": [],
    }

    # --------------------------------------------------
    # Train all models once
    # --------------------------------------------------
    logger.info("Training base models (once, no bootstrap)")

    start = time.time()
    b_twin_models = run_btwin(
        data_params=data_params,
        model_params=b_twin_model_params,
        data=data,
    )
    logger.info("✔ bTwin trained")

    tnet = run_tarnet(data, data_params=data_params)
    logger.info("✔ TARNet trained")

    cfr = run_cfrnet(data, data_params=data_params)
    logger.info("✔ CFRNet trained")

    dnet = run_dragonnet(data, data_params=data_params)
    logger.info("✔ DragonNet trained")

    bnet = run_bcauss(data, data_params=data_params)
    logger.info("✔ BCAUSS trained")

    logger.info(f"Base model training completed in {time.time() - start:.2f}s")

    synctwin = run_synctwin(
        data_params=data_params,
        model_params=synctwin_model_params,
        data=data,
    )
    logger.info("✔ synctwin trained")
    ## compute ATT on original data
    logger.info("Computing ATT estimates on original data")
    att = predict_att_btwin(
        data_params=data_params,
        data=data,
        models=b_twin_models,
    ).item()
    att_estimates["b_twin"].append(att)
    logger.info(f"  b_twin   ATT = {att:.4f}")

    att = predict_att_tarnet(
        model=tnet,
        data_params=data_params,
        data=data,
    ).item()
    logger.info(f"  tnet     ATT = {att:.4f}")
    att_estimates["tnet"].append(att)
    att = predict_att_cfrnet(
        model=cfr,
        data_params=data_params,
        data=data,
    ).item()
    logger.info(f"  cfr      ATT = {att:.4f}")
    att_estimates["cfr"].append(att)
    att = predict_att_dragonnet(
        model=dnet,
        data_params=data_params,
        data=data,
    ).item()
    logger.info(f"  dnet     ATT = {att:.4f}")
    att_estimates["dnet"].append(att)
    att = predict_att_bcauss(
        model=bnet,
        data_params=data_params,
        data=data,
    ).item()
    logger.info(f"  bnet     ATT = {att:.4f}")
    att_estimates["bnet"].append(att)
    att = predict_att_synctwin(
        data_params=data_params,
        new_data=data,
        trained=synctwin,
    ).item()
    att_estimates["synctwin"].append(att)
    logger.info(f"  Synctwin  ATT = {att:.4f}  ")
    # --------------------------------------------------
    # Bootstrap loop
    # --------------------------------------------------
    logger.info("Starting bootstrap loop")

    for i in range(n_bootsraps):
        iter_start = time.time()
        logger.info(f"Bootstrap {i + 1}/{n_bootsraps}")

        treated_group = data[data["treatment"] == 1]
        control_group = data[data["treatment"] == 0]

        treated_bootsrap = select_bootsrap_treated_samples(
            treated_group,
            n_samples=treated_group.id.nunique(),
            seed=i,
        )

        control_bootsrap = select_bootsrap_treated_samples(
            control_group,
            n_samples=control_group.id.nunique(),
            seed=i + 1000,
        )

        df_bootsrap = pd.concat(
            [treated_bootsrap, control_bootsrap],
            axis=0,
        ).reset_index(drop=True)

        logger.debug(
            f"Bootstrap dataset size: {len(df_bootsrap)} "
            f"(treated={len(treated_bootsrap)}, control={len(control_bootsrap)})"
        )

        # ------------------ bTwin ------------------
        b_twin_models = run_btwin(
            data_params=data_params,
            model_params=b_twin_model_params,
            data=df_bootsrap,
        )
        att = predict_att_btwin(
            data_params=data_params,
            data=df_bootsrap,
            models=b_twin_models,
        ).item()
        att_estimates["b_twin"].append(att)
        logger.info(f"  b_twin   ATT = {att:.4f}")

        ## train other models on bootstrap sample

        # ------------------ TARNet ------------------
        tnet = run_tarnet(df_bootsrap, data_params=data_params)

        att = predict_att_tarnet(
            model=tnet,
            data_params=data_params,
            data=df_bootsrap,
        ).item()
        att_estimates["tnet"].append(att)
        logger.info(f"  tnet     ATT = {att:.4f}")

        # ------------------ CFRNet ------------------
        cfr = run_cfrnet(df_bootsrap, data_params=data_params)
        att = predict_att_cfrnet(
            model=cfr,
            data_params=data_params,
            data=df_bootsrap,
        ).item()
        att_estimates["cfr"].append(att)
        logger.info(f"  cfr      ATT = {att:.4f}")

        # ------------------ DragonNet ------------------
        dnet = run_dragonnet(df_bootsrap, data_params=data_params)
        att = predict_att_dragonnet(
            model=dnet,
            data_params=data_params,
            data=df_bootsrap,
        ).item()
        att_estimates["dnet"].append(att)
        logger.info(f"  dnet     ATT = {att:.4f}")

        # ------------------ BCAUSS ------------------
        bnet = run_bcauss(df_bootsrap, data_params=data_params)
        att = predict_att_bcauss(
            model=bnet,
            data_params=data_params,
            data=df_bootsrap,
        ).item()
        att_estimates["bnet"].append(att)
        logger.info(f"  bnet     ATT = {att:.4f}")
        synctwin = run_synctwin(
            data_params=data_params,
            model_params=synctwin_model_params,
            data=df_bootsrap,
        )
        att = predict_att_synctwin(
            trained=synctwin,
            data_params=data_params,
            new_data=df_bootsrap,
        ).item()
        att_estimates["synctwin"].append(att)
        logger.info(f"  synctwin     ATT = {att:.4f}")
        logger.info(f"Bootstrap {i + 1} done in {time.time() - iter_start:.2f}s")

    logger.info("Bootstrap finished")

    att_estimates = pd.DataFrame(att_estimates)

    logger.info("ATT bootstrap summary:")
    logger.info(att_estimates.describe().to_string())

    return att_estimates
