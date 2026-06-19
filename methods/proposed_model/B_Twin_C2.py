import os
import sys
import time

from torch import optim

from .src.ablation import run_att_experiment
from .src.config import (
    DEVICE,
)
from .src.data_format import make_loader
from .src.data_preperation import prepare_data
from .src.model import CausalInferenceModel
from .src.train_utils import train_representations_learner

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from simulations.simulation import generate_simulation


def main(
    params,
    weight_epochs=5000,
    n_experiments=100,
    train_epochs=100,
    batch_size=256,
    n_units_exp=10000,
    save_dir="./results",
    filename="B-Twin_C2_",
    ground_truth=False,
    lr1=1e-3,
):
    data = generate_simulation(**params)
    datasets, configs = prepare_data(data, params)
    train_loader = make_loader(datasets, configs, "train", batch_size)
    val_loader = make_loader(datasets, configs, "val", batch_size)

    # === Initialize and train representation learning model ==
    n_features = 1
    seq_len = 84

    model = CausalInferenceModel(n_features=n_features, seq_len=seq_len)

    model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=lr1)

    t1 = time.time()
    best_model, history = train_representations_learner(
        model, train_loader, val_loader, optimizer, train_epochs, device=DEVICE
    )
    t2 = time.time()

    training_time = t2 - t1
    model = best_model
    # print(f"training_time : {training_time}")

    ## run att experiment
    params["n_units"] = n_units_exp
    run_att_experiment(
        n_experiments, params, model, save_dir, filename, weight_epochs, ground_truth
    )


if __name__ == "__main__":
    main()
