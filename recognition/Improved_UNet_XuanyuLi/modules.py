import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class BasicUNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        num_classes=4,
        base_channels=32
    ):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)

        self.enc1 = DoubleConv(
            in_channels,
            base_channels
        )

        self.enc2 = DoubleConv(
            base_channels,
            base_channels * 2
        )

        self.enc3 = DoubleConv(
            base_channels * 2,
            base_channels * 4
        )

        self.enc4 = DoubleConv(
            base_channels * 4,
            base_channels * 8
        )

        self.bottleneck = DoubleConv(
            base_channels * 8,
            base_channels * 16
        )

        self.up4 = nn.ConvTranspose2d(
            base_channels * 16,
            base_channels * 8,
            kernel_size=2,
            stride=2
        )

        self.dec4 = DoubleConv(
            base_channels * 16,
            base_channels * 8
        )

        self.up3 = nn.ConvTranspose2d(
            base_channels * 8,
            base_channels * 4,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(
            base_channels * 8,
            base_channels * 4
        )

        self.up2 = nn.ConvTranspose2d(
            base_channels * 4,
            base_channels * 2,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(
            base_channels * 4,
            base_channels * 2
        )

        self.up1 = nn.ConvTranspose2d(
            base_channels * 2,
            base_channels,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(
            base_channels * 2,
            base_channels
        )

        self.output = nn.Conv2d(
            base_channels,
            num_classes,
            kernel_size=1
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        x = self.bottleneck(self.pool(e4))

        x = self.up4(x)
        x = torch.cat((x, e4), dim=1)
        x = self.dec4(x)

        x = self.up3(x)
        x = torch.cat((x, e3), dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat((x, e2), dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat((x, e1), dim=1)
        x = self.dec1(x)

        return self.output(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dropout=0.0
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        if dropout > 0:
            self.dropout = nn.Dropout2d(dropout)
        else:
            self.dropout = nn.Identity()

        # Channel numbers must match before residual addition.
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = x + shortcut
        x = self.relu(x)

        return x


class ImprovedUNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        num_classes=4,
        base_channels=32
    ):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)

        self.enc1 = ResidualBlock(
            in_channels,
            base_channels
        )

        self.enc2 = ResidualBlock(
            base_channels,
            base_channels * 2
        )

        self.enc3 = ResidualBlock(
            base_channels * 2,
            base_channels * 4,
            dropout=0.1
        )

        self.enc4 = ResidualBlock(
            base_channels * 4,
            base_channels * 8,
            dropout=0.2
        )

        self.bottleneck = ResidualBlock(
            base_channels * 8,
            base_channels * 16,
            dropout=0.3
        )

        self.up4 = nn.ConvTranspose2d(
            base_channels * 16,
            base_channels * 8,
            kernel_size=2,
            stride=2
        )

        self.dec4 = ResidualBlock(
            base_channels * 16,
            base_channels * 8
        )

        self.up3 = nn.ConvTranspose2d(
            base_channels * 8,
            base_channels * 4,
            kernel_size=2,
            stride=2
        )

        self.dec3 = ResidualBlock(
            base_channels * 8,
            base_channels * 4
        )

        self.up2 = nn.ConvTranspose2d(
            base_channels * 4,
            base_channels * 2,
            kernel_size=2,
            stride=2
        )

        self.dec2 = ResidualBlock(
            base_channels * 4,
            base_channels * 2
        )

        self.up1 = nn.ConvTranspose2d(
            base_channels * 2,
            base_channels,
            kernel_size=2,
            stride=2
        )

        self.dec1 = ResidualBlock(
            base_channels * 2,
            base_channels
        )

        self.output = nn.Conv2d(
            base_channels,
            num_classes,
            kernel_size=1
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        x = self.bottleneck(self.pool(e4))

        x = self.up4(x)
        x = torch.cat((x, e4), dim=1)
        x = self.dec4(x)

        x = self.up3(x)
        x = torch.cat((x, e3), dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat((x, e2), dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat((x, e1), dim=1)
        x = self.dec1(x)

        return self.output(x)


def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


if __name__ == "__main__":
    test_input = torch.randn(
        2,
        1,
        128,
        128
    )

    baseline = BasicUNet()
    improved = ImprovedUNet()

    baseline_output = baseline(test_input)
    improved_output = improved(test_input)

    print(
        "Basic U-Net output:",
        baseline_output.shape
    )

    print(
        "Improved U-Net output:",
        improved_output.shape
    )

    print(
        "Basic U-Net parameters:",
        count_parameters(baseline)
    )

    print(
        "Improved U-Net parameters:",
        count_parameters(improved)
    )