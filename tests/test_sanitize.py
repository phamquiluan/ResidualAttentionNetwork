import torch
import resattnet
from resattnet import resattnet56


def test_sanitize():
    m = resattnet56(in_channels=3, num_classes=10, pretrained=False)  # pretrained is load automatically

    tensor = torch.Tensor(1, 3, 224, 224)

    output = m(tensor)

    print(output.shape)  # torch.Size([1, 10])


test_sanitize()
