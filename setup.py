#!/usr/bin/env python
import os
from setuptools import find_packages, setup

version = "0.0.1"
cwd = os.path.dirname(os.path.abspath(__file__))


def write_version_file():
    version_path = os.path.join(cwd, "resattnet", "version.py")
    with open(version_path, "w") as f:
        f.write("__version__ = '{}'\n".format(version))


write_version_file()

setup(
    name="ResisualAttentionNetwork",
    description="Pre-trained Residual Attention Network",
    version=version,
    packages=find_packages(exclude=["models"]),
    include_package_data=True,
    install_requires=[
        "torch",
        "torchvision",
        "pytorchcv",
        "tqdm",
        "natsort",
        "imgaug",
        "tensorboard",
        "sklearn",
        "gputil"
    ],
)
