from dacapo_toolbox.transforms.weight_balancing import BalanceLabels

import torch
import pytest


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
