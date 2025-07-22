import torch

class DistanceHead(torch.nn.Module):
    def __init__(self, in_channels: int = 12, incr_factor: int = 12, dims: int = 3):
        super().__init__()
        self.dims = dims
        self.conv1 = torch.nn.Conv3d(
            in_channels, in_channels * incr_factor, kernel_size=1
        )
        self.relu = torch.nn.LeakyReLU()
        self.conv2 = torch.nn.Conv3d(in_channels * incr_factor, 1, kernel_size=1)
        self.sigmoid = torch.nn.Sigmoid()

        torch.nn.init.kaiming_normal_(
            self.conv1.weight, mode="fan_out", nonlinearity="relu"
        )
        torch.nn.init.kaiming_normal_(
            self.conv2.weight, mode="fan_out", nonlinearity="sigmoid"
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = self.sigmoid(self.conv2(self.relu(self.conv1(x - y))))
        # remove channel dimension
        if out.ndim == self.dims + 1:
            out = out.unsqueeze(0)
        elif out.ndim == self.dims + 2:
            out = out.squeeze(1)
        return out