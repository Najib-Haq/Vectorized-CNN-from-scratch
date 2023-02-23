import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import os
import sys

from dataset.dataset import Dataset, DataLoader, check_dataset
from model.Model import Model
from utils import set_seed, split_dataset, wandb_init, mixup
from fit import *


if __name__ == "__main__":
    set_seed(42)

    # read config
    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)
    os.makedirs(config['output_dir'], exist_ok=True)
    wandb_run = wandb_init(config) if config['use_wandb'] else None
    config['wandb_id'] = wandb_run.id if config['use_wandb'] else None
    
    # make model
    model = Model(config)
    print(model)
    if config['checkpoint_path']:
        model.load_model(config['checkpoint_path'], pretrained=True)

    # make dataset
    train_df, valid_df = split_dataset(config['data_dir'], validation_percentage=0.2)
    print("Train: ", train_df.shape, "; Valid: ", valid_df.shape)
    
    if config['debug']:
        train_df = train_df.sample(frac=1).reset_index(drop=True)[:100]
        valid_df = valid_df.sample(frac=1).reset_index(drop=True)[:100]

    train_dataset = Dataset(config['data_dir'], train_df, label_col='digit', mode='train', config=config['augment'])
    valid_dataset = Dataset(config['data_dir'], valid_df, label_col='digit', mode='valid', config=config['augment'])
    check_dataset(train_dataset, valid_dataset, save_dir=config['output_dir'])

    train_loader = DataLoader(train_dataset, batch_size=config['train_batch'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['valid_batch'], shuffle=False)
    if config['augment']['mixup']:
        data = next(iter(train_loader))
        images, labels = data[0], data[1]
        images, labels = mixup(images, labels, num_classes=config['num_class'])

        check_dataset([images, labels], valid_dataset, save_dir=config['output_dir'], from_mixup=True)

    # train
    fit_model(model, train_loader, valid_loader, config, wandb_run)


