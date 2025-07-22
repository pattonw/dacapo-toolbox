import torch

import itertools


def balance_weights(
    labels: torch.Tensor,
    mask: torch.Tensor | None = None,
    num_classes: int = 2,
    slab=None,
    clipmin: float = 0.05,
    clipmax: float = 0.95,
):
    if labels.dtype in [torch.uint8, torch.uint16, torch.uint32, torch.uint64]:
        # torch doesn't support many operations on unsigned ints
        labels = labels.long()

    unique_labels = torch.unique(labels)
    assert len(unique_labels) <= num_classes, (
        f"Found unique labels {unique_labels} but expected only {num_classes}."
    )
    assert 0 <= labels.min() < num_classes, (
        f"Labels {unique_labels} are not in [0, {num_classes})."
    )
    assert 0 <= labels.min() < num_classes, (
        f"Labels {unique_labels} are not in [0, {num_classes})."
    )

    # initialize error scale with 1s
    error_scale = torch.ones(labels.shape, dtype=torch.float32)

    # set error_scale to 0 in masked-out areas
    if mask is not None:
        error_scale = error_scale * mask
    else:
        mask = torch.ones_like(labels, dtype=bool)

    if slab is None:
        slab = error_scale.shape
    else:
        # slab with -1 replaced by shape
        slab = tuple(m if s == -1 else s for m, s in zip(error_scale.shape, slab))

    slab_ranges = (range(0, m, s) for m, s in zip(error_scale.shape, slab))

    for start in itertools.product(*slab_ranges):
        slices = tuple(slice(start[d], start[d] + slab[d]) for d in range(len(slab)))
        # operate on slab independently
        scale_slab = error_scale[slices]
        labels_slab = labels[slices]
        # in the masked-in area, compute the fraction of per-class samples
        masked_in = scale_slab.sum()
        classes, counts = torch.unique(
            labels_slab[mask[slices] > 0], return_counts=True
        )
        classes = classes.long()  # ensure classes are long for indexing
        fracs = (
            counts.float() / masked_in if masked_in > 0 else torch.zeros(counts.shape)
        )
        if clipmin is not None or clipmax is not None:
            fracs = torch.clip(fracs, clipmin, clipmax)

        # compute the class weights
        total_frac = 1.0
        w_sparse = total_frac / float(num_classes) / fracs
        w = torch.zeros(num_classes)
        w[classes] = w_sparse

        # scale_slab the masked-in scale_slab with the class weights
        scale_slab *= torch.take(w, labels_slab.long())
        error_scale[slices] = scale_slab

    return error_scale


class BalanceLabels(torch.nn.Module):
    def __init__(
        self,
        slab=None,
        num_classes: int = 2,
        clipmin: float = 0.05,
        clipmax: float = 0.95,
    ):
        super().__init__()
        self.slab = slab
        self.num_classes = num_classes
        self.clipmin = clipmin
        self.clipmax = clipmax

    def forward(self, labels, mask=None):
        return balance_weights(
            labels, mask, self.num_classes, self.slab, self.clipmin, self.clipmax
        )
