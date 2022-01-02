# -*- coding: utf-8 _*_
# @Time : 29/12/2021 4:59 pm
# @Author: ZHA Mengyue
# @FileName: visualization.py
# @Software: TimeSeriesAutoencoder
# @Blog: https://github.com/Dolores2333

from components import *

from matplotlib import pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def pca_and_tsne(args):
    ori_ts = np.load(args.ori_data_dir)   # (len_ori_data, seq_len, z_dim)
    art_ts = np.load(args.art_data_dir)   # (len_ori_data, seq_len, z_dim)
    len_ori_data = len(ori_ts)
    subplots = [231, 232, 233, 234, 235, 236]

    # Plot PCA
    plt.figure(args.samples_to_plot, figsize=(16, 12))
    for k in range(min(args.z_dim, 6)):
        ts_ori_k = ori_ts[..., k]  # (len_ori_data, seq_len)
        ts_art_k = art_ts[..., k]  # (len_ori_data, seq_len)

        pca = PCA(n_components=2)
        pca.fit(ts_ori_k)
        pca_ori = pca.transform(ts_ori_k)
        pca_art = pca.transform(ts_art_k)

        plt.subplot(subplots[k])
        plt.grid()
        plot_scatter(pca_ori[:, 0], pca_ori[:, 1], color='b', alpha=0.1)
        plot_scatter(pca_art[:, 0], pca_art[:, 1], color='r', alpha=0.1)
        plt.title(f'PCA plots for {args.columns[k]}')
        plt.xlabel('x-pca')
        plt.ylabel('y-pca')
        file_name = f'{args.instance_name}_pca.png'
        file_dir = os.path.join(args.pics_dir, file_name)
        plt.savefig(file_dir)

    # Plot t-SNE
    plt.figure(args.samples_to_plot + 1, figsize=(16, 12))
    for k in range(min(args.z_dim, 6)):
        ts_ori_k = ori_ts[..., k]  # (len_ori_data, seq_len)
        ts_art_k = art_ts[..., k]  # (len_ori_data, seq_len)
        ts_final_k = np.concatenate((ts_ori_k, ts_art_k), axis=0)

        tsne = TSNE(n_components=2)
        tsne_results = tsne.fit_transform(ts_final_k)

        plt.subplot(subplots[k])
        plt.grid()
        plot_scatter(tsne_results[:len_ori_data, 0],
                     tsne_results[:len_ori_data, 1],
                     color='b', alpha=0.1, label='Original')
        plot_scatter(tsne_results[len_ori_data:, 0],
                     tsne_results[len_ori_data:, 1],
                     color='r', alpha=0.1, label='Synthetic')
        plt.legend()
        plt.title(f't-SNE plots for {args.columns[k]}')
        plt.xlabel('x-tsne')
        plt.ylabel('y-tsne')
        file_name = f'{args.instance_name}_tsne.png'
        file_dir = os.path.join(args.pics_dir, file_name)
        plt.savefig(file_dir)

    # Plot PCA and t-SNE in TimeGAN style
    ori_ts = np.mean(ori_ts, axis=-1, keepdims=False)  # (len_ori_data, seq_len)
    art_ts = np.mean(art_ts, axis=-1, keepdims=False)  # (len_ori_data, seq_len)
    tsne_ts = np.concatenate((ori_ts, art_ts), axis=0)   # (2 * len_ori_data, seq_len)

    plt.figure(args.samples_to_plot + 2, figsize=(16, 12))

    pca = PCA(n_components=2)
    pca.fit(ori_ts)
    pca_ori = pca.transform(ori_ts)
    pca_art = pca.transform(art_ts)

    plt.subplot(121)
    plt.grid()
    plot_scatter(pca_ori[:, 0], pca_ori[:, 1], color='b', alpha=0.1)
    plot_scatter(pca_art[:, 0], pca_art[:, 1], color='r', alpha=0.1)
    plt.title(f'PCA plots for features averaged')
    plt.xlabel('x-pca')
    plt.ylabel('y-pca')

    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(tsne_ts)

    plt.subplot(122)
    plt.grid()
    plot_scatter(tsne_results[:len_ori_data, 0],
                 tsne_results[:len_ori_data, 1],
                 color='b', alpha=0.1, label='Original')
    plot_scatter(tsne_results[len_ori_data:, 0],
                 tsne_results[len_ori_data:, 1],
                 color='r', alpha=0.1, label='Synthetic')
    plt.legend()
    plt.title(f't-SNE plots for features averaged')
    plt.xlabel('x-tsne')
    plt.ylabel('y-tsne')

    file_name = f'{args.instance_name}_visualization.png'
    file_dir = os.path.join(args.pics_dir, file_name)
    plt.savefig(file_dir)


# plot utils
def plot_scatter(*args, **kwargs):
    # plt.plot(*args, **kwargs)
    plt.scatter(*args, **kwargs)


def plot_time_series_no_masks(args):
    ori_ts = np.load(args.ori_data_dir)   # (len_ori_data, seq_len, z_dim)
    art_ts = np.load(args.art_data_dir)  # (len_ori_data, seq_len, z_dim)
    for i in range(args.samples_to_plot):
        ts_ori = ori_ts[i]  # (seq_len, z_dim)
        ts_art = art_ts[i]  # (seq_len, z_dim)

        subplots = [231, 232, 233, 234, 235, 236]
        plt.figure(i, figsize=(16, 12))

        # Plot max 6 features
        for k in range(min(args.z_dim, 6)):
            ts_ori_k = ts_ori[:, k]  # (seq_len, )
            ts_art_k = ts_art[:, k]  # (seq_len, )
            plt.subplot(subplots[k])
            plt.grid()
            plot_scatter(range(args.ts_size), ts_ori_k, color='b')
            plot_scatter(range(args.ts_size), ts_art_k, color='g')
            plt.title(f'{args.columns[k]}')
        file_dir = os.path.join(args.pics_dir, f'sample{i}.png')
        plt.savefig(file_dir)


def plot_time_series_with_masks(args):
    """ori blue, art_mask red art_no_mask green"""
    ori_ts = np.load(args.ori_data_dir)   # (len_ori_data, seq_len, z_dim)
    art_ts = np.load(args.art_data_dir)  # (len_ori_data, seq_len, z_dim)
    masks = np.load(args.masks_dir)  # (len_ori_data, seq_len)
    for i in range(args.samples_to_plot):
        ts_ori = ori_ts[i]  # (seq_len, z_dim)
        ts_art = art_ts[i]  # (seq_len, z_dim)
        mask = masks[i]  # (seq_len, )

        subplots = [231, 232, 233, 234, 235, 236]
        plt.figure(i, figsize=(16, 12))

        # Plot max 6 features
        for k in range(min(args.z_dim, 6)):
            ts_ori_k = ts_ori[:, k]  # (seq_len, )
            ts_art_k = ts_art[:, k]  # (seq_len, )
            ts_art_k_mask = ts_art_k[mask]
            ts_art_k_no_mask = ts_art_k[~mask]
            plt.subplot(subplots[k])
            plt.grid()
            plot_scatter(range(args.ts_size), ts_ori_k, color='b')
            plot_scatter([j for j in range(args.ts_size) if mask[j]], ts_art_k_mask, color='r')
            plot_scatter([j for j in range(args.ts_size) if not mask[j]], ts_art_k_no_mask, color='g')
            plt.title(f'{args.columns[k]}')
        file_dir = os.path.join(args.pics_dir, f'sample{i}.png')
        plt.savefig(file_dir)


if __name__ == '__main__':
    home = os.getcwd()
    args = load_arguments(home)
    file_dir = os.path.join(args.model_dir, 'args_dict.npy')
    args_dict = load_dict_npy(file_dir)[()]
    # print(type(args_dict))
    args = argparse.Namespace(**args_dict)
    pca_and_tsne(args)
