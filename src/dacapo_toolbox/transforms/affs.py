import torch
from collections.abc import Sequence
import itertools
from typing import Callable


def compute_affs(
    arr: torch.Tensor,
    offset: Sequence[int],
    dist_func: Callable,
    pad: bool = False,
) -> torch.Tensor:
    """
    Compute affinities on a given tensor `arr` using the specified `offset` and distance
    function `dist_func`. if `pad` is True, `arr` will be padded s.t. the output shape
    matches the input shape.
    """
    offset: torch.Tensor = torch.tensor(offset, device=arr.device)
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
        dist_func: str | Callable | list[Callable] = "equality",
        pad: bool = True,
        concat_dim: int = 0,
    ):
        super(Affs, self).__init__()
        self.neighborhood = neighborhood
        self.ndim = len(neighborhood[0])
        self.pad = pad
        self.concat_dim = concat_dim
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
            try:
                dist_iterator = iter(dist_func)  # type: ignore
            except TypeError:
                raise ValueError(f"Unknown distance function: {dist_func}")
            iterable_dist_func = list(dist_iterator)
            if all(isinstance(func, torch.nn.Module) for func in iterable_dist_func):
                self.dist_func = torch.nn.ModuleList(iterable_dist_func)
            elif all(callable(func) for func in iterable_dist_func):
                self.dist_func = iterable_dist_func
            else:
                raise ValueError(f"Unknown distance function: {dist_func}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not isinstance(self.dist_func, torch.nn.ModuleList)) and callable(
            self.dist_func
        ):
            dist_funcs = [self.dist_func] * len(self.neighborhood)
        else:
            dist_funcs = self.dist_func

        affs = [
            compute_affs(x, offset, dist_func, pad=self.pad)
            for offset, dist_func in zip(self.neighborhood, dist_funcs)
        ]
        if self.pad:
            out = torch.stack(affs, dim=self.concat_dim)
        else:
            # Find minimum shape along each axis
            shapes = [torch.tensor(aff.shape) for aff in affs]
            min_shape = torch.stack(shapes).min(dim=0).values.tolist()

            # Crop all tensors to min_shape
            affs = [aff[tuple(slice(0, s) for s in min_shape)] for aff in affs]
            out = torch.stack(affs, dim=self.concat_dim)
        return out


class AffsMask(torch.nn.Module):
    def __init__(
        self,
        neighborhood: Sequence[Sequence[int]],
        pad=True,
    ):
        super(AffsMask, self).__init__()
        self.neighborhood = neighborhood
        self.dist_func = no_bg_dist_func
        self.pad = pad

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        y = mask.int() > 0
        affs = [
            compute_affs(y, offset, self.dist_func, pad=self.pad)
            for offset in self.neighborhood
        ]
        if self.pad:
            return torch.stack(affs, dim=0)
        else:
            # Find minimum shape along each axis
            min_shape = (
                torch.stack([torch.tensor(aff.shape) for aff in affs])
                .min(dim=0)
                .values.tolist()
            )

            # Crop all tensors to min_shape
            affs = [aff[tuple(slice(0, s) for s in min_shape)] for aff in affs]
            return torch.stack(affs)
