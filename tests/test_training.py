from data import corrupt_mnist
import torch
import os
from pathlib import Path

from omegaconf import OmegaConf

from src.train import train
from src.model import MyAwesomeModel
import pytest

def test_training(tmp_path, monkeypatch):
    # --- minimal config to keep test fast ---
    cfg = OmegaConf.create(
        {
            "model": {
                "conv1": 8,
                "conv2": 16,
                "conv3": 32,
                "dropout": 0.1,
                "num_classes": 10,
            },
            "training": {
                "lr": 1e-3,
                "batch_size": 32,
                "epochs": 1,
                "seed": 123,
            },
        }
    )

    train(cfg)

    assert Path("models/model.pth").exists()

def test_error_on_wrong_shape():
    model = MyAwesomeModel(
        conv1=8,
        conv2=8,
        conv3=8,
        dropout=0.5,
        num_classes=10
    )
    with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
        model(torch.randn(1,2,3))
    with pytest.raises(
        ValueError,
        match=r"Expected each sample to have shape \[1, 28, 28\]",
    ):
        model(torch.randn(1, 1, 28, 29))