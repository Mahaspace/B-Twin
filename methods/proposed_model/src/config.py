import torch

D_TYPE = torch.float32
DEVICE = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
# params = {
#     "n_units": 10000,
#     "n_time": 168,
#     "treatment_time": 84,
#     "phi": 0.8,
#     "sigma": 5,
#     "alpha": 0.05,
#     "seed": None,
#     "bias": True,
#     "constant_effect": True,
# }
