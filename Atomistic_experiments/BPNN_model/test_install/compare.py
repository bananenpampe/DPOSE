import torch
import os

os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_pred_energy.pt")

A = torch.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_pred_energy.pt"))
B = torch.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "materials_model_predictions", "BaTiO3", "shallow_ens_nll", "test_pred_energy.pt",))

assert torch.allclose(A, B[:10]), "reference and redicted values are not close."