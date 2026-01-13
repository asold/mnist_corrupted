import matplotlib.pyplot as plt
import torch
from data import corrupt_mnist
from pathlib import Path
import hydra
import logging
from omegaconf import DictConfig
from torch import nn

class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(
        self,
        conv1: int,
        conv2: int,
        conv3: int,
        dropout: float,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, conv1, 3, 1)
        self.conv2 = nn.Conv2d(conv1, conv2, 3, 1)
        self.conv3 = nn.Conv2d(conv2, conv3, 3, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(conv3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError('Expected input to a 4D tensor')
        if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
            raise ValueError('Expected each sample to have shape [1, 28, 28]')

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

log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


@hydra.main(version_base="1.1", config_path="../conf", config_name="config")
def train(cfg: DictConfig) -> None:

    torch.manual_seed(cfg.training.seed)

    Path("models").mkdir(parents=True, exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)


    log.info("Configuration:\n%s", cfg)

    model = MyAwesomeModel(
        conv1=cfg.model.conv1,
        conv2=cfg.model.conv2,
        conv3=cfg.model.conv3,
        dropout=cfg.model.dropout,
        num_classes=cfg.model.num_classes,
    ).to(DEVICE)

    train_set, _ = corrupt_mnist()

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.training.batch_size
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(cfg.training.epochs):
        model.train()
        for i, (img, target) in enumerate(train_loader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    log.info("Training complete")
    torch.save(model.state_dict(), "models/model.pth")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("reports/figures/training_statistics.png")


if __name__ == "__main__":
    train()
