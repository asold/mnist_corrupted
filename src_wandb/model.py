import torch
from torch import nn


class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(
        self,
        conv1_out: int,
        conv2_out: int,
        conv3_out: int,
        dropout: float,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, conv1_out, 3, 1)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, 3, 1)
        self.conv3 = nn.Conv2d(conv2_out, conv3_out, 3, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(conv3_out, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc1(x)


if __name__ == "__main__":
    model = MyAwesomeModel()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
