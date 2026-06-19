import torch

D_TYPE = torch.float32
DEVICE = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")

params_exp = {
    "seed": "100",
    "model_id": "",
    "itr": "1",
    "itr_pretrain": "5000",
    "itr_fine_tune": "2000",
    "batch_size": "100",
    "pretrain_Y": "False",
    "reduced_fine_tune": "True",
    "linear_decoder": "False",
    "lam_prognostic": "1",
    "lam_recon": "1",
    "tau": "1",
    "n_hidden": "20",
    "sim_id": None,
    "regular": "True",
}

## global simulation params
simulation_params = {
    "n_units": 10000,
    "n_time": 168,
    "treatment_time": 84,
    "phi": 0.8,
    "sigma": 5,
    "alpha": 0.05,
    "seed": None,
    "bias": True,
    "constant_effect": True,
}
