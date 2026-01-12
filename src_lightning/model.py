import pytorch_lightning as pl
import torch
from torch import nn
import wandb


class MyAwesomeModel(pl.LightningModule):
    """My awesome model."""

    def __init__(
        self,
        conv1_out: int,
        conv2_out: int,
        conv3_out: int,
        dropout: float,
        num_classes: int,
        lr: float,
    ):
        super().__init__()
        self.save_hyperparameters()  # Lightning best practice

        self.conv1 = nn.Conv2d(1, conv1_out, 3, 1)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, 3, 1)
        self.conv3 = nn.Conv2d(conv2_out, conv3_out, 3, 1)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(conv3_out, num_classes)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)
    
    def training_step(self, batch):
        """Training step."""
        img, target = batch
        y_pred = self(img)
        return self.loss_fn(y_pred, target)

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def training_step(self, batch, batch_idx):
        img, target = batch
        preds = self(img)
        loss = self.loss_fn(preds, target)
        acc = (preds.argmax(dim=1) == target).float().mean()

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        # log non-scalar data every N steps
        if batch_idx % 100 == 0:
            self.logger.experiment.log(
                {"logits": wandb.Histogram(preds.detach().cpu())}
            )

        return loss
    
    def validation_step(self, batch, batch_idx):
        img, target = batch
        preds = self(img)
        loss = self.loss_fn(preds, target)
        acc = (preds.argmax(dim=1) == target).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)




if __name__ == "__main__":
    model = MyAwesomeModel()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")