import matplotlib.pyplot as plt
import torch
from data import corrupt_mnist
from src.model import MyAwesomeModel
from pathlib import Path
import hydra
import logging
from omegaconf import DictConfig

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
