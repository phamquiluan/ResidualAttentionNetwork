import os
import torch
import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model


def resattnet56(in_channels=3, num_classes=1000, pretrained=True):
    model = ptcv_get_model("resattnet56", pretrained=False)

    if pretrained is True:
        state = torch.hub.load_state_dict_from_url(
            "https://github.com/phamquiluan/ResidualAttentionNetwork/releases/download/v0.1.0/resattnet56.pth"
        )
        model.load_state_dict(state["state_dict"])
    model.output = nn.Linear(2048, num_classes)
    return model
