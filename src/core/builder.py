import torch
import torch.nn as nn
from typing import List
from torch.types import Device
from torch.optim.optimizer import Optimizer

from .datasets import *
from .models import *
from utils import *

def make_datasets(configs: Config,
                  mode: str,
                  train_path=None,
                  val_path=None,
                  test_path=None,
                  transforms=None,
                  separator=' ',
                  device="cpu"):

    # Generate Dataset
    if configs.datasets.type == 'on_gpu':
        dataset = OnGPUDataset(
            mode=mode,
            dataset_dir=configs.datasets.dir,
            input_data_format_postfix=configs.datasets.input_type,
            output_data_format_postfix=configs.datasets.output_type,
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            transforms=transforms,
            separator=separator,
            device=device,
        )
    else:
        raise NotImplementedError

    return dataset

def make_transforms(configs: Config):
    transforms = []
    # add one more dimension
    transforms.append(Add_Dimension())
    if configs.transforms.label_normalization:
        transforms.append(Label_Normalization())
    
    return transforms

def make_model(configs: Config):
    if configs.model.input_dim == configs.model.output_dim:
        in_downsample = False
    elif configs.model.input_dim == 2 * configs.model.output_dim:
        in_downsample = True

    if configs.model.name == "unet" :
        model = UNet(
            configs.model.in_channels,
            configs.model.mid_channels,
            configs.model.out_channels,
            in_downsample
        )
    else:
        raise NotImplementedError
    
    return model


def make_optimizer(
    params,
    configs: Config
) -> Optimizer:

    if configs.optimizer.name == "adam":
        optimizer = torch.optim.Adam(
            params,
            configs.optimizer.lr,
            weight_decay=configs.optimizer.weight_decay,
        )
    elif configs.optimizer.name == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            configs.optimizer.lr,
            weight_decay=configs.optimizer.weight_decay,
        )
    else:
        raise NotImplementedError(configs.optimizer.name)

    return optimizer

def make_lr_scheduler(
    configs: Config,
    optimizer: Optimizer,
):
    if configs.lr_scheduler.name == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)
    else:
        raise NotImplementedError()
    return scheduler

def make_criterion(
    configs: Config
):
    if configs.criterion.name == 'mse':
        criterion = nn.MSELoss()
    elif configs.criterion.name  == 'huber':
        criterion = nn.HuberLoss(delta=configs.criterion.delta)
    else:
        raise NotImplementedError()
    return criterion


