import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model


def resattnet56(in_channels, num_classes, pretrained=True):
    model = ptcv_get_model("resattnet56", pretrained=False)

    if pretrained is True:
        # load pretrained automatically
        pass

    model.output = nn.Linear(2048, num_classes)
    return model
