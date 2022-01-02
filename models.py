# -*- coding: utf-8 _*_
# @Time : 28/12/2021 7:59 pm
# @Author: ZHA Mengyue
# @FileName: models.py
# @Software: TimeSeriesAutoencoder
# @Blog: https://github.com/Dolores2333

from tqdm import tqdm
from visualization import *


class AutoEncoderUnit(nn.Module):
    def __init__(self, args):
        super(AutoEncoderUnit, self).__init__()
        self.args = args
        self.ts_size = args.ts_size
        self.z_dim = args.z_dim
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

    def forward(self, x):
        # x(-1, ts_size, z_dim)
        # x = posenc(x)
        x_enc = self.encoder(x)  # (-1, ts_size, hidden_dim)
        # x_enc = posenc(x_enc)
        x_dec = self.decoder(x_enc)  # (-1, ts_size, z_dim)
        return x_dec


class AutoEncoder(nn.Module):
    def __init__(self, args, ori_data):
        super(AutoEncoder, self).__init__()
        self.args = args
        self.device = torch.device(args.device)
        self.ori_data = ori_data
        self.model = AutoEncoderUnit(args).to(self.device)
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.results = {'n_updates': 0,
                        'loss': []}
        print(f'Successfully initialized {self.__class__.__name__}!')

    def train_ae(self):
        self.model.train()

        for t in tqdm(range(self.args.ae_epochs)):
            x_ori = get_batch(args=self.args, data=self.ori_data)
            x_ori = torch.tensor(x_ori, dtype=torch.float32).to(self.device)
            x_hat = self.model(x_ori)
            loss = self.criterion(x_hat, x_ori)

            self.results['n_updates'] = t
            self.results['loss'].append(loss.clone().detach().cpu().numpy())
            if t % self.args.log_interval == 0:
                print(f'Epoch {t} with {loss.item()} loss')
                if bool(self.args.save):
                    save_model(self.args, self.model)
                    save_metrics_results(self.args, self.results)
                    save_args(self.args)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def evaluate_ae(self):
        """Evaluate the model as a simple AntoEncoder"""
        self.model.eval()
        ori_data = torch.tensor(self.ori_data, dtype=torch.float32).to(self.device)
        art_data = self.model(ori_data)
        art_data = art_data.clone().detach().cpu().numpy()
        art_data *= self.args.max_val
        art_data += self.args.min_val
        np.save(self.args.art_data_dir, art_data)  # save art_data after renormalization
        plot_time_series_no_masks(self.args)
        pca_and_tsne(self.args)
