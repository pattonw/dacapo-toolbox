import torch
from collections.abc import Sequence
import itertools
from typing import Callable


def compute_affs(
    arr: torch.Tensor,
    offset: Sequence[int],
    dist_func: callable,
    pad: bool = False,
) -> torch.Tensor:
    """
    Compute affinities on a given tensor `arr` using the specified `offset` and distance
    function `dist_func`. if `pad` is True, `arr` will be padded s.t. the output shape
    matches the input shape.
    """
    offset = torch.tensor(offset, device=arr.device)
    offset_dim = len(offset)

    if pad:
        padding = itertools.chain(
            *(
                (0, axis_offset) if axis_offset > 0 else (-axis_offset, 0)
                for axis_offset in list(offset)[::-1]
            )
        )
        arr = torch.nn.functional.pad(arr, tuple(padding), mode="constant", value=0)

    arr_shape = arr.shape[-offset_dim:]
    slice_ops_lower = tuple(
        slice(
            max(0, -offset[h]),
            min(arr_shape[h], arr_shape[h] - offset[h]),
        )
        for h in range(0, offset_dim)
    )
    slice_ops_upper = tuple(
        slice(
            max(0, offset[h]),
            min(arr_shape[h], arr_shape[h] + offset[h]),
        )
        for h in range(0, offset_dim)
    )

    # handle arbitrary number of leading dimensions (can be batch, channel, etc.)
    # distance function should handle batch/channel dimensions appropriately
    while len(slice_ops_lower) < len(arr.shape):
        slice_ops_lower = (slice(None), *slice_ops_lower)
        slice_ops_upper = (slice(None), *slice_ops_upper)

    return dist_func(
        arr[slice_ops_lower],
        arr[slice_ops_upper],
    )

def equality_dist_func(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x == y

def equality_no_bg_dist_func(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x == y) * (x > 0) * (y > 0)

def no_bg_dist_func(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x > 0) * (y > 0)

class Affs(torch.nn.Module):
    def __init__(
        self,
        neighborhood: Sequence[Sequence[int]],
        dist_func: str | Callable = "equality",
    ):
        super(Affs, self).__init__()
        self.neighborhood = neighborhood
        self.ndim = len(neighborhood[0])
        assert all(len(offset) == self.ndim for offset in neighborhood), (
            "All offsets in the neighborhood must have the same dimensionality."
        )
        if dist_func == "equality":
            self.dist_func = equality_dist_func
        elif dist_func == "equality-no-bg":
            self.dist_func = equality_no_bg_dist_func
        elif callable(dist_func):
            self.dist_func = dist_func
        else:
            raise ValueError(f"Unknown distance function: {dist_func}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                compute_affs(x, offset, self.dist_func, pad=True)
                for offset in self.neighborhood
            ],
            dim=0,
        )


class AffsMask(torch.nn.Module):
    def __init__(self, neighborhood: Sequence[Sequence[int]]):
        super(AffsMask, self).__init__()
        self.neighborhood = neighborhood
        self.dist_func = no_bg_dist_func

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.ones_like(x, dtype=torch.bool)
        return torch.stack(
            [
                compute_affs(y, offset, self.dist_func, pad=True)
                for offset in self.neighborhood
            ],
            dim=0,
        )
