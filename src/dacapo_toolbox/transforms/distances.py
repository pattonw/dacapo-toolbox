import torch
import numpy as np
from edt import edt


class SignedDistanceTransform(torch.nn.Module):
    """
    Computes the signed distance transform of a label mask.
    The output is normalized to the range [-1, 1] using `tanh(dist/sigma)`.
    """

    def __init__(self, sigma: float = 10.0):
        super(SignedDistanceTransform, self).__init__()
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.numpy()
        return torch.nn.functional.tanh(
            torch.from_numpy(edt(x) - edt(x == 0)) / self.sigma
        )


class SDTBoundaryMask(torch.nn.Module):
    """
    Computes a binary mask of regions where the distance to the nearest boundary
    is less than the distance to the border of the image. This is useful to avoid
    training on the ambiguous boundary regions where the true distance to the nearest
    object is not known.
    """

    def __init__(self, sigma: float = 10.0):
        super(SDTBoundaryMask, self).__init__()
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.numpy()
        b = np.ones_like(x)
        return torch.from_numpy(edt(x) + edt(x == 0) < edt(b, black_border=True))
