import pytest
import torch

from dacapo_toolbox.transforms.affs import Affs, AffsMask
from dacapo_toolbox.transforms.distances import SDTBoundaryMask, SignedDistanceTransform
from dacapo_toolbox.transforms.lsds import LSD
from dacapo_toolbox.transforms.weight_balancing import BalanceLabels

LABELS = torch.tensor(
    [
        [1, 1, 2],
        [1, 0, 2],
        [3, 3, 0],
    ]
)
# one channel per offset: a voxel against its neighbour to the right, and below
NEIGHBORHOOD = [(0, 1), (1, 0)]


@pytest.mark.parametrize(
    "options, expected",
    [
        # padded with background, so the last column and row meet label 0
        ({}, [[[1, 0, 0], [0, 0, 0], [1, 0, 1]], [[1, 0, 1], [0, 0, 0], [0, 0, 1]]]),
        (
            {"dist_func": "equality-no-bg"},
            [[[1, 0, 0], [0, 0, 0], [1, 0, 0]], [[1, 0, 1], [0, 0, 0], [0, 0, 0]]],
        ),
        ({"pad": False}, [[[1, 0], [0, 0]], [[1, 0], [0, 0]]]),
        (
            {"dist_func": [torch.eq, torch.ne]},
            [[[1, 0, 0], [0, 0, 0], [1, 0, 1]], [[0, 1, 0], [1, 1, 1], [1, 1, 0]]],
        ),
    ],
    ids=["equality", "equality-no-bg", "unpadded", "one-function-per-offset"],
)
def test_affs(options, expected):
    affs = Affs(NEIGHBORHOOD, **options)(LABELS)
    assert affs.int().tolist() == expected


def test_affs_concat_dim_places_the_offset_channels():
    batch = torch.stack([LABELS, LABELS])
    assert Affs(NEIGHBORHOOD, concat_dim=1)(batch).shape == (2, 2, 3, 3)


@pytest.mark.parametrize(
    "pad, expected",
    [
        (True, [[[1, 1, 0], [0, 0, 0], [1, 0, 0]], [[1, 0, 1], [1, 0, 0], [0, 0, 0]]]),
        (False, [[[1, 1], [0, 0]], [[1, 0], [1, 0]]]),
    ],
    ids=["padded", "unpadded"],
)
def test_affs_mask_is_set_between_two_foreground_voxels(pad, expected):
    with pytest.warns(DeprecationWarning, match="AffsMask is deprecated"):
        affs_mask = AffsMask(NEIGHBORHOOD, pad=pad)
    assert affs_mask(LABELS).int().tolist() == expected


def test_signed_distance_transform():
    # distance to the nearest voxel of the other kind, negative in the
    # background, squashed by tanh(distance / sigma)
    mask = torch.tensor([0, 0, 1, 1, 1], dtype=torch.uint8)
    distances = SignedDistanceTransform(sigma=10.0)(mask)
    torch.testing.assert_close(
        distances, torch.tanh(torch.tensor([-2.0, -1.0, 1.0, 2.0, 3.0]) / 10.0)
    )


def test_sdt_boundary_mask():
    # set where the object boundary (2, 1, 1, 2, 3 voxels away) is nearer than
    # the border of the image (1, 2, 3, 2, 1 voxels away)
    mask = torch.tensor([0, 0, 1, 1, 1], dtype=torch.uint8)
    assert SDTBoundaryMask()(mask).tolist() == [False, True, True, False, False]


@pytest.mark.parametrize("dims", [2, 3])
@pytest.mark.parametrize(
    "options",
    [{}, {"downsample": 2}, {"voxel_size": 2, "sigma": 4.0}],
    ids=["full-resolution", "downsampled", "voxel-size-2"],
)
def test_lsd_deep_inside_an_object_and_in_the_background(dims, options):
    # one cube of 24 voxels in a volume of 32. Around its centre the object
    # fills the whole gaussian window, so the descriptor is that of the
    # gaussian itself: no offset to the centre of mass (0.5), a variance of
    # sigma squared along each axis (1 after normalising; the window is cut
    # at 3 sigma and sampled on voxels, hence the tolerance), no covariance
    # between axes (0) and full mass (1). The background has no descriptor.
    options = {"sigma": 2.0, **options}
    if "voxel_size" in options:
        options["voxel_size"] = (options["voxel_size"],) * dims
    segmentation = torch.zeros((32,) * dims, dtype=torch.int64)
    segmentation[(slice(4, 28),) * dims] = 5
    off_diagonal = dims * (dims - 1) // 2

    descriptors = LSD(**options)(segmentation)

    assert descriptors.shape == (2 * dims + off_diagonal + 1, *segmentation.shape)
    torch.testing.assert_close(
        descriptors[(slice(None), *(16,) * dims)],
        torch.tensor([0.5] * dims + [1.0] * dims + [0.0] * off_diagonal + [1.0]),
        atol=0.03,
        rtol=0,
    )
    assert (descriptors[:, segmentation == 0] == 0).all()


@pytest.mark.parametrize("masked", [True, False])
@pytest.mark.parametrize(
    "dtype",
    [
        int,
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ],
)
def test_balance_weights(masked, dtype):
    labels = torch.tensor(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
        ],
        dtype=dtype,
    )
    if masked:
        mask = torch.tensor(
            [
                [1, 1, 1, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 0],
                [0, 0, 0, 0],
            ],
            dtype=torch.uint8,
        )
        expected_label_weights = [4 / 9 * 3, 2 / 9 * 3, 3 / 9 * 3]
        expected_weights = mask[:].float()
        for label, weight in enumerate(expected_label_weights):
            expected_weights[labels == label] /= weight
    else:
        mask = None
        expected_label_weights = [4 / 16 * 3, 4 / 16 * 3, 8 / 16 * 3]
        expected_weights = torch.ones_like(labels).float()
        for label, weight in enumerate(expected_label_weights):
            expected_weights[labels == label] /= weight

    weights = BalanceLabels(num_classes=3)(labels, mask=mask)

    torch.testing.assert_close(weights, expected_weights)
