# -*- coding: utf-8 _*_
# @Time : 27/12/2021 2:15 pm
# @Author: ZHA Mengyue
# @FileName: utils.py
# @Software: TimeSeriesAutoencoder
# @Blog: https://github.com/Dolores2333


import os
import json
import math
import argparse
import numpy as np
import pandas as pd
from einops import rearrange

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt

"""utils.py includes
    1. Args Loading
    2. Data Related
    3. Model Related"""

"""Args Loading"""


def load_arguments(home):
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--config', default='config.json')
    args_dict = vars(parser.parse_args())
    args_dict['home'] = home
    config_name = args_dict['config']
    print(f'Loaded config named {config_name}')
    config_dir = os.path.join(home, config_name)

    with open(config_dir, 'r') as f:
        config_dict = json.load(fp=f)

    total_dict = {**config_dict, **args_dict}
    # Find dirs and stores them into the args
    storage_dir = os.path.join(home, 'storage')
    if not os.path.isdir(storage_dir):
        os.mkdir(storage_dir)

    instance_dir = os.path.join(storage_dir, total_dict['instance_name'])
    model_dir = os.path.join(instance_dir, 'model')
    pics_dir = os.path.join(instance_dir, 'pics')
    if not os.path.isdir(instance_dir):
        os.mkdir(instance_dir)
        os.mkdir(model_dir)
        os.mkdir(pics_dir)

    total_dict['storage_dir'] = storage_dir
    total_dict['instance_dir'] = instance_dir
    total_dict['model_dir'] = model_dir
    total_dict['pics_dir'] = pics_dir
    datasets_dir = os.path.join(home, 'data')
    total_dict['datasets_dir'] = datasets_dir
    # dirs for specific dataset
    total_dict['stock_dir'] = os.path.join(datasets_dir, 'stock_data.csv')
    total_dict['energy_dir'] = os.path.join(datasets_dir, 'energy_data.csv')
    # total_dict['working_dir'] = os.path.join(datasets_dir, 'working_data.npy')

    art_data_dir = os.path.join(model_dir, 'art_data.npy')
    ori_data_dir = os.path.join(model_dir, 'ori_data.npy')
    masks_dir = os.path.join(model_dir, 'masks.npy')
    total_dict['art_data_dir'] = art_data_dir
    total_dict['ori_data_dir'] = ori_data_dir
    total_dict['masks_dir'] = masks_dir

    args = argparse.Namespace(**total_dict)

    json_dir = os.path.join(args.model_dir, 'instance.json')
    os.system(f'cp {config_dir} {json_dir}')

    return args


"""Data Related"""


def min_max_scalar(data):
    """Min-Max Normalizer.

    Args:
      - data: raw data

    Returns:
      - norm_data: normalized data
      - min_val: minimum values (for renormalization)
      - max_val: maximum values (for renormalization)
    """
    min_val = np.min(np.min(data, axis=0), axis=0)  # (z_dim, ) min for each feature
    data = data - min_val

    max_val = np.max(np.max(data, axis=0), axis=0)  # (z_dim, ) max for each feature
    norm_data = data / (max_val + 1e-7)

    return norm_data, min_val, max_val


def sine_data_generation(num_samples, seq_len, z_dim):
    """Sine data generation
       Remark: no args.min/max/var for sine_data
               no normalization
               no renormalization
    Args:
        - num_samples: the number of samples
        - seq_len: the sequence length of the time-series
        - dim: feature dimensions
    Returns:
        - data: generated data"""
    sine_data = list()
    for i in range(num_samples):
        single_sample = list()
        for k in range(z_dim):
            # Randomly drawn frequency and phase for each feature (column)
            freq = np.random.uniform(low=0, high=0.1)
            phase = np.random.uniform(low=0, high=0.1)
            sine_feature = [np.sin(freq * j + phase) for j in range(seq_len)]
            single_sample.append(sine_feature)
        single_sample = np.transpose(np.asarray(single_sample))  # (seq_len, z_dim)
        single_sample = (single_sample + 1) * 0.5
        # Stack the generated data
        sine_data.append(single_sample)
    sine_data = np.array(sine_data)  # (num_sample, seq_len, z_dim)
    return sine_data


def sliding_window(args, ori_data):
    """ Slicing the ori_data by sliding window
        Args:
            args
            ori_data (len(csv), z_dim)
        Returns:
            ori_data (:, seq_len, z_dim)"""
    # Flipping the data to make chronological data
    ori_data = ori_data[::-1]  # (len(csv), z_dim)
    # Make (len(ori_data), z_dim) into (num_samples, seq_len, z_dim)
    samples = []
    for i in range(len(ori_data)-args.ts_size):
        single_sample = ori_data[i:i + args.ts_size]  # (seq_len, z_dim)
        samples.append(single_sample)
    samples = np.array(samples)  # (bs, seq_len, z_dim)
    np.random.shuffle(samples)  # Make it more like i.i.d.
    return samples


def load_data(args):
    """Load and preprocess rea-world datasets and record necessary statistics
    Args:
        - data_name: stock or energy
        - seq_len: sequence length
    Returns:
        - data: preprocessed data"""
    assert args.data_name in ['stock', 'energy', 'sine']
    ori_data = None

    if args.data_name == 'stock':
        ori_data = np.loadtxt(args.stock_dir, delimiter=',', skiprows=1)
        ori_data = sliding_window(args, ori_data)
        args.columns = pd.read_csv(args.stock_dir).columns
    elif args.data_name == 'energy':
        ori_data = np.loadtxt(args.energy_dir, delimiter=',', skiprows=1)
        ori_data = sliding_window(args, ori_data)
        args.columns = pd.read_csv(args.energy_dir).columns
    elif args.data_name == 'sine':
        ori_data = sine_data_generation(num_samples=10000, seq_len=args.ts_size, z_dim=args.z_dim)
        args.columns = [f'feature{i}' for i in range(args.z_dim)]

    # saving the processed data for work under args.working_dir
    np.save(args.ori_data_dir, ori_data)
    return ori_data


def get_batch(args, data):
    idx = np.random.permutation(len(data))
    idx = idx[:args.batch_size]
    data_mini = data[idx, ...]  # (bs, seq_len, z_dim)
    return data_mini


"""Model Related"""


def save_model(args, model):
    file_dir = os.path.join(args.model_dir, 'model.pth')
    torch.save(model.state_dict(), file_dir)


def save_metrics_results(args, results):
    file_dir = os.path.join(args.model_dir, 'metrics_results.npy')
    np.save(file_dir, results)


def save_args(args):
    file_dir = os.path.join(args.model_dir, 'args_dict.npy')
    np.save(file_dir, args.__dict__)


def load_model(args, model):
    model_dir = args.model_dir
    file_dir = os.path.join(model_dir, 'model.pth')

    model_state_dict = torch.load(file_dir)
    model.load_state_dict({f'model.{k}': v for k, v in model_state_dict.items()})
    return model


def load_dict_npy(file_path):
    file = np.load(file_path, allow_pickle=True)
    return file
