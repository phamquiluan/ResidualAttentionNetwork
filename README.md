# ImageNet training in PyTorch - Residual Attention Network

[![version](https://img.shields.io/badge/version-v0.0.1-blue)](https://github.com/phamquiluan/ResidualAttentionNetwork)
[![phamquiluan/ResidualAttentionNetwork](https://circleci.com/gh/phamquiluan/ResidualAttentionNetwork.svg?style=shield&circle-token=f96e4e1a66e86406f9a01512c52e1185b731ab0e)](https://app.circleci.com/pipelines/github/phamquiluan/ResidualAttentionNetwork)


This implements training of [Residual Attention Network](https://arxiv.org/abs/1704.06904) on the ImageNet dataset, and provide the pretrained weights.


## Pretrained Download

Download resattnet56 pretrained Imagenet1K: [link](https://drive.google.com/file/d/1Sc-TCERxrJKN4TvmDOwn_98GeUva_FIr/view?usp=sharing)

Eval: Acc@1 77.024 Acc@5 93.574


## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)
## Training

To train a model, run `main.py` with the desired model architecture and the path to the ImageNet dataset:

```bash
python main.py -a resattnet56 [imagenet-folder with train and val folders]
```

## Multi-processing Distributed Data Parallel Training

You should always use the NCCL backend for multi-processing distributed training since it currently provides the best distributed training performance.

### Single node, multiple GPUs:

```bash
python main.py -a resattnet56 --dist-url 'tcp://127.0.0.1:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 [imagenet-folder with train and val folders]
```
