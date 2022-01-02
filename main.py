# -*- coding: utf-8 _*_
# @Time : 28/12/2021 4:33 pm
# @Author: ZHA Mengyue
# @FileName: main.py
# @Software: TimeSeriesAutoencoder
# @Blog: https://github.com/Dolores2333


from models import *


def run_ae():
    home = os.getcwd()
    args = load_arguments(home)

    # Data Loading
    ori_data = load_data(args)
    np.save(args.ori_data_dir, ori_data)  # save ori_data before normalization
    ori_data, min_val, max_val = min_max_scalar(ori_data)
    # Write statistics
    args.min_val = min_val
    args.max_val = max_val
    args.data_var = np.var(ori_data)
    print(f'{args.data_name} data variance is {args.data_var}')

    # Initialize teh Model
    model = AutoEncoder(args, ori_data)
    if args.training:
        print('Start AutoEncoder Training!')
        model.train_ae()
        print('AutoEncoder Training Finished!')
    else:
        model = load_model(args, model)
        print('Successfully loaded the model!')

    print('Start Evaluation.')
    model.evaluate_ae()
    print('Evaluation Finished!')


if __name__ == "__main__":
    run_ae()
    """Test AutoEncoderUnit Directly
    # Args
    home = os.getcwd()
    args = load_arguments(home)

    # Data Loading
    ori_data = load_data(args)
    np.save(args.ori_data_dir, ori_data)  # save ori_data before normalization
    ori_data, min_val, max_val = min_max_scalar(ori_data)
    # Write statistics
    args.min_val = min_val
    args.max_val = max_val
    args.data_var = np.var(ori_data)
    print(f'{args.data_name} data variance is {args.data_var}')

    # Define model, loss and optimizer
    model = AutoEncoderUnit(args)
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters())
    results = {'n_updates': 0,
               'loss': []}

    # Model training
    model.train()
    for t in range(args.ae_epochs):
        x_ori = torch.tensor(get_batch(args, ori_data), dtype=torch.float32)
        x_hat = model(x_ori)
        loss = criterion(x_hat, x_ori)

        results['n_updates'] = t
        results['loss'].append(loss.clone().detach().cpu().numpy())
        if t % args.log_interval == 0:
            print(f'Epoch {t} with {loss.item()} loss')
            if bool(args.save):
                save_model(args, model)
                save_metrics_results(args, results)
                save_args(args)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
                                           
    # Model evaluation
    model.eval()
    ori_data = torch.tensor(ori_data, dtype=torch.float32)
    art_data = model(ori_data)
    art_data = art_data.clone().detach().cpu().numpy()
    art_data *= max_val
    art_data += min_val
    np.save(args.art_data_dir, art_data)  # save art_data after renormalization

    plot_time_series_no_masks(args, 10)
    """
