"""Timm model for 2D image classification."""

from typing import Literal, Callable

import torch
import torch.nn as nn
import timm


class Timm(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, model_name: str):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, in_chans=in_channels, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
