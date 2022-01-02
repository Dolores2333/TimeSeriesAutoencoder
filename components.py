# -*- coding: utf-8 _*_
# @Time : 27/12/2021 1:56 pm
# @Author: ZHA Mengyue
# @FileName: components.py
# @Software: TimeSeriesAutoencoder
# @Blog: https://github.com/Dolores2333

import os
import numpy as np
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.cluster.vq import kmeans2

from utils import *


def get_sinusoid_encoding_table(n_position, d_model):
    def get_position_angle_vector(position):
        exponent = [2 * (j // 2) / d_model for j in range(d_model)]  # [d_model,]
        position_angle_vector = position / np.power(10000, exponent)  # [d_model,]
        return position_angle_vector
    sinusoid_table = np.array([get_position_angle_vector(i) for i in range(n_position)])
    # [0::2]: 2i, [1::2]: 2i+1
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    # table of size (n_position, d_model)
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


def posenc(x):
    # x(bs, seq_len, z_dim)
    b , l, f = x.shape
    position_encoding = get_sinusoid_encoding_table(l, f)
    position_encoding = position_encoding.type_as(x).to(x.device).clone().detach()
    x += position_encoding
    return x


def mask_and_posenc(args, x, masks):
    """Add position encoding and Split x in to x_visible and x_masked"""
    # x(bs, seq_len, z_dim)
    b, l, f = x.shape
    position_encoding = get_sinusoid_encoding_table(args.ts_size, args.z_dim)
    position_encoding = position_encoding.type_as(x).to(x.device).clone().detach()
    x += position_encoding
    x_visible = x[~masks, :].reshape(b, -1, f)  # (bs, -1, z_dim)
    return x_visible


def posenc_and_concatenate(args, x_visible, masks):
    """Add position encoding adn concatenate x_visible and x_masked"""
    b, l, f = x_visible.shape  # (bs, -1, embed_dim)
    # print(f'Shape of x_visible is {x_visible.shape}')
    x_masked = nn.Parameter(torch.zeros(b, args.ts_size, f))[masks, :].reshape(b, -1, f)
    # print(f'Shape o x_masked id {x_masked.shape}')
    position_encoding = get_sinusoid_encoding_table(args.ts_size, f)
    position_encoding = position_encoding.expand(b, -1, -1).type_as(x_visible).to(x_visible.device).clone().detach()
    # print(f'Shape of position encoding is {position_encoding.shape}')
    # print(f'Shape masks is {masks.shape}')
    visible_position_encoding = position_encoding[~masks, :].reshape(b, -1, f)
    masked_position_encoding = position_encoding[masks, :].reshape(b, -1, f)
    x_visible += visible_position_encoding
    x_masked += masked_position_encoding
    x_full = torch.cat([x_visible, x_masked], dim=1)  # x_full(bs, seq_len, embed_dim)
    return x_full


def mask_only(args, x, masks):
    # x(bs, seq_len, z_dim)
    b, l, f = x.shape
    x_visible = x[~masks, :].reshape(b, -1, f)  # (bs, -1, z_dim)
    return x_visible


def mask_only_with_masks(args, x, masks):
    # x(bs, seq_len, z_dim)
    b, l, f = x.shape
    x[masks, :] = torch.normal(mean=0, std=1, size=(1,))
    return x


def concatenate_only(args, x_visible, masks):
    b, l, f = x_visible.shape  # (bs, -1, hidden_dim)
    # x_masked = nn.Parameter(torch.zeros(b, args.ts_size, f))[masks, :].reshape(b, -1, f)
    x_masked = nn.Parameter(torch.normal(mean=0, std=1, size=(b, args.ts_size, f)))[masks, :].reshape(b, -1, f)
    x_full = torch.cat([x_visible, x_masked], dim=1)  # x_full (bs, seq_len, hidden_dim)
    return x_full


def reshuffle_only(args, x_visible, masks):
    b, l, f = x_visible.shape  # (bs, -1, hidden_dim)
    x_full = nn.Parameter(torch.zeros(b, args.ts_size, f))
    x_full[masks, :] = x_visible
    return x_full


"""Masked Auto Encoder Components based on RNN
    1. Encoder
    2. Decoder
    3. Quantize"""


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.rnn = nn.RNN(input_size=args.z_dim,
                          hidden_size=args.hidden_dim,
                          num_layers=args.num_layer)
        self.fc = nn.Linear(in_features=args.hidden_dim,
                            out_features=args.hidden_dim)

    def forward(self, x):
        x_enc, _ = self.rnn(x)
        x_enc = self.fc(x_enc)
        return x_enc


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.rnn = nn.RNN(input_size=args.hidden_dim,
                          hidden_size=args.hidden_dim,
                          num_layers=args.num_layer)
        self.fc = nn.Linear(in_features=args.hidden_dim,
                            out_features=args.z_dim)

    def forward(self, x_enc):
        x_dec, _ = self.rnn(x_enc)
        x_dec = self.fc(x_dec)
        return x_dec


class VectorQuantize(nn.Module):
    """Quantinizes the continuous embedding vector to nearest ones in the codebook
    Args:
        - z (bs, seq_len, hidden_dim) is the output of encoder

    Returns:
        z_q (bs, seq_len, embed_dim) is the quantized embedding vecto
        diff: kld_scale * (2nd + 3rd terms in teh loss)
        ind (bs, seq_len): indices for vector selection from the codebook
                            for each position in (bs, seq_len)"""
    def __init__(self, args):
        super(VectorQuantize, self).__init__()
        self.device = torch.device(args.device)
        self.embed_dim = args.embed_dim
        self.num_embed = args.num_embed
        self.beta = args.beta
        self.kld_scale = 10.0
        self.embed = nn.Embedding(self.num_embed, self.embed_dim)
        self.register_buffer('data_initialized', torch.zeros(1))

    def codebook(self, z_q_idx):
        return F.embedding(z_q_idx, self.embed.weight)

    def forward(self, z_e):
        batch_size, seq_len, hidden_dim = z_e.size() # ze (bs, seq_len, embed_dim)
        flatten = z_e.reshape(-1, self.embed_dim)  # (bs * seq_len, embed_dim)

        # Initialization
        if self.training and self.data_initialized.item() == 0:
            print('Initialized by K-menas')
            random_points = torch.randperm(n=flatten.size(0))
            random_idx = random_points[:2000]
            centroids, labels = kmeans2(data=flatten[random_idx].data.cpu().numpy(),
                                        k=self.num_embed,
                                        minit='points')
            # each row in centroids are a centroid
            centroids = torch.from_numpy(centroids).to(self.device)
            self.embed.weight = nn.Parameter(centroids, requires_grad=True)
            # self.embed.weight = nn.Parameter(torch.from_numpy(centroids),
            #                                  requires_grad=True).to(self.device)  # (num_embed, embed_dim)
            self.data_initialized.fill_(1)
            print('K-means initialized successfully!')

        # Pairwise dist between embedding vectors and vectors in the codebook
        # (bs * seq_len, 1) - 2 * (bs * seq_len, embed_dim) * (embed_dim, num_embed) + (1, num_embed)
        # dist is like (bs * seq_lens, num_embed)
        dist = (flatten.pow(2).sum(dim=1, keepdim=True)
                - 2 * flatten @ self.embed.weight.t()
                + self.embed.weight.pow(2).sum(dim=1, keepdim=True).t())

        # Find teh max and corresponding idx for each row
        row_max, max_idx = (-dist).max(1)
        z_q_idx = max_idx.view(batch_size, seq_len)

        z_q = self.codebook(z_q_idx)  # (bs, seq_len, embed_dim)
        diff = (z_q.detach() - z_e).pow(2).mean() + self.beta * (z_q - z_e.detach()).pow(2).mean()
        diff *= self.kld_scale

        z_q = z_e + (z_q - z_e).detach()
        z_q = rearrange(z_q, 'b l e -> b e l')
        return z_q, diff, z_q_idx
