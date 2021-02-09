import torch
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import seaborn as sns
sns.set_theme()
import glob
import re



import torch.optim
import torch
import argparse


#from torchvision import models, datasets, transforms

try:
    from tqdm import tqdm
except:
    def tqdm(x): return x


def process_df(quant, dirname, args=None, args_model=None, save=True):

    idx = pd.IndexSlice
    losses = quant.loc[:, idx[:, 'loss']]
    errors = quant.loc[:, idx[:, 'error']]

    if save:
        quant.to_csv(os.path.join(dirname, 'quant.csv'))

    losses.to_csv(os.path.join(dirname, 'losses.csv'))
    errors.to_csv(os.path.join(dirname, 'errors.csv'))

    losses.describe().to_csv(os.path.join(dirname, 'losses_describe.csv'))
    errors.describe().to_csv(os.path.join(dirname, 'errors_describe.csv'))

    fig=plt.figure()
    df_reset = quant.reset_index()
    df_plot = pd.melt(df_reset, id_vars='draw')
    g = sns.relplot(
        data = df_plot,
        #col='',
        #hue='set',
        col='stat',
        x='layer',
        y='value',
        kind='line',
        ci=100,
        #col_wrap=2,
        facet_kws={
            'sharey': False,
            'sharex': True
        }
    )
    g.fig.subplots_adjust(top=0.9, left=1/g.axes.shape[1] * 0.1)
    if args_model is not None and args is not None:
        width  = args_model.width
        if width is None:
            if args_model.dataset == "mnist":
                width = 245  # WARNING hard coded
        removed = "width / {}".format(args.fraction) if hasattr(args, 'fraction') and args.fraction is not None else args.remove
        g.fig.suptitle('ds = {}, width = {}, removed = {}, draw = {}'.format(args_model.dataset, width, removed, args.ndraw))
    g.set(yscale='linear')
    plt.savefig(fname=os.path.join(dirname, 'plot.pdf'))
    g.set(yscale='log')
    plt.savefig(fname=os.path.join(dirname, 'plot_log.pdf'))


    return

def process_checkpoint(checkpoint):
    '''Read and process a previously computed result stored inside a checkpoint (for the copy test)'''

    quant = checkpoint['quant']
    args = checkpoint['args']
    idx = pd.IndexSlice
    quant.loc[:, idx[:, 'error']] *= 100  # in %
    process_df(quant, args.path_output)
    return

def process_csv(file_csv):
    '''Read and process a previously computed result stored inside a checkpoint'''

    idx = pd.IndexSlice
    quant = pd.read_csv(file_csv, header=[0,1], index_col=0)
    #quant.loc[:, idx[:, 'error']] *= 100  # in percent

    dirname = os.path.dirname(file_csv)
    process_df(quant, dirname, save=False)
    return



def process_subdir(subdir, device, N_L=5, N_T=20):
    # subdir will have different entry_n results, all with the same number of
    # removed units

    # 1. process all the entry_n files
    # 2. store the results in a bundle dataframe (do not forget the different
    # epochs)
    # 3. save / plot the resulting bundle dataframe
    regex_entry = re.compile("entry_(\d+)")
    layers = np.arange(1, N_L+1)#classifier.n_layers)  # the different layers, forward order
    stats = ['loss', 'error']
    #tries = np.arange(1, 1+args.ndraw)  # the different tries

    names=['set', 'layer', 'stat']
    columns=pd.MultiIndex.from_product([layers, stats], names=names)
    #index = pd.Index(np.arange(1, start_epoch+args.nepoch+1), name='epoch')
    index = pd.Index(np.arange(1, N_T+1), name='draw')
    df_bundle = pd.DataFrame(columns=columns, index=index, dtype=float)
    epochs = {}

    df_bundle.sort_index(axis=1, inplace=True)  # sort for quicker access
    Idx = pd.IndexSlice


    for file_entry in glob.glob(os.path.join(subdir, "checkpoint_entry_*.pth"), recursive=False):
        #match = regex_entry.search(file_entry)
        #if match is None:
        #    continue
        checkpoint = torch.load(file_entry, map_location=device)
        idx_entry = checkpoint['args'].entry_layer#int(match.groups()[0])
        if idx_entry > 0:
            args = checkpoint['args']
        epoch = checkpoint['epochs']
        quant = checkpoint['quant']

        df_bundle = pd.concat([df_bundle, quant], ignore_index=False, axis=1)

        #df_bundle.loc[Idx[:, (idx_entry,'loss')]] = quant.loc[Idx[epoch, ('train', 'loss')]]
        #df_bundle.loc[Idx[:, (idx_entry,'error')]] = quant.loc[Idx[epoch, ('train', 'error')]]
        epochs[idx_entry] = epoch

    df_bundle.sort_index(axis=1, inplace=True)  # sort for quicker access
    return df_bundle, epochs, args



if __name__ == '__main__':

    torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser('Evaluating a copy of a classifier with removed units')
    #parser.add_argument('--dataset', '-dat', default='mnist', type=str, help='dataset')
    #parser.add_argument('--dataroot', '-droot', default='./data/', help='the root for the input data')
    parser.add_argument('--name', default='eval-copy', type=str, help='the name of the experiment')
    #parser.add_argument('--vary_name', nargs='*', default=None, help='the name of the parameter to vary in the name (appended)')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3, help='leraning rate')
    parser.add_argument('--lr_update', type=int, default=0, help='update for the learning rate in epochs')
    parser.add_argument('--save_model', action='store_true', default=True, help='stores the model after some epochs')
    parser.add_argument('--nepoch', type=int, default=1000, help='the number of epochs to train for')
    parser.add_argument('--batch_size', '-bs', type=int, default=100, help='the dimension of the batch')
    parser.add_argument('--debug', action='store_true', help='debug')
    parser.add_argument('--size_max', type=int, default=None, help='maximum number of traning samples')
    parser.add_argument('--ndraw', type=int, default=10, help='The number of permutations to test')
    parser.add_argument('-R', '--remove', type=int, default=100, help='the number of neurons to remove at each layer')
    parser_model = parser.add_mutually_exclusive_group(required=False)
    parser_model.add_argument('--model', help='path of the model to separate')
    parser_model.add_argument('--checkpoint', help='path of the previous computation checkpoint')
    parser_model.add_argument('--csv', help='path of the previous saved csv file')
    parser.add_argument('--gd_mode', '-gdm', default='stochastic', choices=['full', 'stochastic'], help='whether the gradient is computed full batch or stochastically')
    parser_device = parser.add_mutually_exclusive_group()
    parser_device.add_argument('--cpu', action='store_true', dest='cpu', help='force the cpu model')
    parser_device.add_argument('--cuda', action='store_false', dest='cpu')
    parser.add_argument('--depth_max', type=int, help='the maximum depth to which operate')
    #parser.add_argument('--end_layer', type=int, help='if set the maximum layer for which to compute the separation (forward indexing)')
    parser.add_argument('--draw', type=int, default=10, help='The number of draw to take')
    parser.set_defaults(cpu=False)
    parser.add_argument('dir', nargs='*', help='the directory to process')



    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    #device = torch.device('cpu')


    dtype = torch.float
    num_gpus = torch.cuda.device_count()

    for d in args.dir:
        args_model = args = None
        try:
            args_model = torch.load(os.path.join(os.path.dirname(d.rstrip(os.sep)), "checkpoint.pth"), map_location=device)['args']
        except :
            pass
        df, epochs, args_entry = process_subdir(d, device)
        process_df(df, d, args_entry, args_model)


