import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from data import corrupt_mnist
from model import MyAwesomeModel

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping



log = logging.getLogger(__name__)
log = logging.getLogger(__name__)


@hydra.main(version_base="1.1", config_path="../conf", config_name="config")
def train(cfg: DictConfig) -> None:
    # reproducibility
    torch.manual_seed(cfg.training.seed)

    # output dirs (Lightning will also create its own)
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    log.info("Configuration:\n%s", cfg)

    # ------------------------
    # Data
    # ------------------------
    train_set, val_set = corrupt_mnist()

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.training.batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg.training.batch_size,
    )

    # ------------------------
    # Model (LightningModule)
    # ------------------------
    model = MyAwesomeModel(
        conv1=cfg.model.conv1,
        conv2=cfg.model.conv2,
        conv3=cfg.model.conv3,
        dropout=cfg.model.dropout,
        num_classes=cfg.model.num_classes,
        lr=cfg.training.lr,
    )

    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-model",
    )

    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min",
    )


    # ------------------------
    # Logger
    # ------------------------
    wandb_logger = WandbLogger(
        project="dtu_mlops",
        log_model=True,
    )

    # ------------------------
    # Trainer (STEP 4)
    # ------------------------
    trainer = Trainer(
        max_epochs=cfg.training.epochs,          # replaces manual epoch loop
        limit_train_batches=0.2,                  # 20% of training data
        callbacks=[checkpoint_cb, early_stop_cb],
        default_root_dir="models",                # checkpoints & logs
        logger=wandb_logger,
        accelerator="auto",                       # CPU / CUDA / MPS automatically
        devices="auto",
    )

    # ------------------------
    # Train
    # ------------------------
    trainer.fit(model, train_loader, val_loader)

    log.info("Training complete")


if __name__ == "__main__":
    train()