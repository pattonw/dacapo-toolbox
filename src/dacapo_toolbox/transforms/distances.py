import torch
import numpy as np
from edt import edt


class SignedDistanceTransform(torch.nn.Module):
    def __init__(self, sigma=10.0):
        super(SignedDistanceTransform, self).__init__()
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.numpy()
        return torch.nn.functional.tanh(
            torch.from_numpy(edt(x) - edt(x == 0)) / self.sigma
        )


class SDTBoundaryMask(torch.nn.Module):
    def __init__(self, sigma=10.0):
        super(SDTBoundaryMask, self).__init__()
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.numpy()
        b = np.ones_like(x)
        return torch.from_numpy(edt(x) + edt(x == 0) < edt(b, black_border=True))
