import torch
import numpy as np
import pandas as pd
import os
import sys
from torchsummary import summary
import torch.nn as nn
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import seaborn as sns
sns.set_theme()
sns.set(font_scale=3, rc={'text.usetex' : False})
sns.set_theme()
sns.set_style('whitegrid')
import glob
import re
import pdb

import math

import models
import random

import torch.optim
import torch
import argparse
import utils

from sklearn.linear_model import LogisticRegression

#from torchvision import models, datasets, transforms

try:
    from tqdm import tqdm
except:
    def tqdm(x): return x


def process_epochs(epochs, dirname):

    fig = plt.figure()
    columns = pd.Index(range(0, len(epochs)), name='layer')
    df = pd.DataFrame(epochs, index=['epoch'], columns=columns)
    df = df.melt()
    s = df.plot(x='layer', y='value', kind='scatter', ylabel='epoch')
    s.set(ylabel="epoch")
    plt.savefig(fname=os.path.join(dirname, 'epochs.pdf'))
    return

def select_min(df):
    """Select the test with the minimal error (usually 0)"""

    Idx = pd.IndexSlice
    df_min = None
    n_layers = len(df.columns.levels[0])
    #columns = df.columns.name
    indices = np.zeros(n_layers, dtype=int)


    for idx in range(n_layers):
        # replace NaN with 0
        val_min = df.loc[:, (idx, 'error')].min()
        mask = df.loc[:, (idx, 'error')] == val_min
        indices[idx] = df.loc[mask, (idx, 'loss')].idxmin()  # if several min, take the min of them
        # the indices for the try that has the minimum training
        # error at the epoch epoch

    # remove the column index 'try'
    cols = pd.MultiIndex.from_product(df.columns.levels, names=df.columns.names)  # all but the try
    df_min = pd.DataFrame(columns=cols, index=[1])
    df_min.index.name = 'step'

    for idx in range(n_layers):
        # select the try that has the minimum training error at the
        # last epoch (at index indices[idx])
        df_min.loc[1, Idx[idx, :]] = df.loc[indices[idx],Idx[idx, :]]#.droplevel('try', axis=1)
    #df_min.loc[:, df_min.columns.get_level_values('layer') == 'last'] = df.xs(('last', idx_last), axis=1, level=[2, 3], drop_level=False).droplevel('try', axis=1)
    #df_min.loc[:, df_min.columns.get_level_values('stat') == 'err'] *= 100
    #df_min = df_min.loc[pd.IndexSlice[:, df_min.columns.get_level_values('layer').isin(range(1, n_layers+1))]]

    #if not df.loc[epoch,  ('train', 'err', 1, indices[0] )] == 0:
    #    print('Not separated!', dirname)
    #else:
    return df_min
    #    print('Separated!', dirname)


def process_df(quant, dirname, stats_ref=None, args=None, args_model=None, save=True):

    global table_format
    idx = pd.IndexSlice
    #losses = quant.loc[:, idx[:, '#loss']]
    #errors = quant.loc[:, idx[:, 'error']]

    #col_order = ["layer", "set", "stat"]
    col_order = ["stat", "set", "layer"]
    if quant.columns.names != col_order:
        # the order is
        # perform pivot
        quant = pd.melt(quant.reset_index(), id_vars="var").pivot(index="var", columns=col_order, values="value")
    idx_order = ["stat", "set"]
    if stats_ref.index.names !=idx_order:
        stats_ref = stats_ref.reorder_levels(idx_order).sort_index(axis=0)

    quant_describe = quant.groupby(level=["stat", "set"], axis=1, group_keys=False).describe()
    if save:
        quant.to_csv(os.path.join(dirname, 'quant.csv'))
        if stats_ref is not None:
            stats_ref.to_csv(os.path.join(dirname, 'stats_ref.csv'))
        quant_describe.to_csv(os.path.join(dirname, 'describe.csv'))



    # table_err_train = table["err"]["train"]
    #quant.loc[:, Idx[:, :, "err"]] *= 100
    if len(stats_ref.keys()) == 1:
        stats_ref = stats_ref[stats_ref.keys()[0]]
    #quant["err"] *= 100
    #stats_ref_copy  = stats_ref.copy()
    #stats_ref_copy["err"] = stats_ref["err"] * 100
    stats_ref.sort_index(axis=0, inplace=True)
    quant.sort_index(axis=1, inplace=True)
    #losses.to_csv(os.path.join(dirname, 'losses.csv'))
    #errors.to_csv(os.path.join(dirname, 'errors.csv'))
    N_L = len(quant.columns.unique(level="layer")) # number of layers
    #N_sets = len(quant.columns.unique(level="set"))
    N_sets=2   # only train and test
    palette=sns.color_palette(n_colors=N_sets)

    #losses.describe().to_csv(os.path.join(dirname, 'losses_describe.csv'))
    df_reset = quant.reset_index()
    #relative quantities
    #N_L = len(quant.columns.unique(level="layer")) -1 # number of hidden layers
    N_S = len(stats_ref)
    stats_ref_val = stats_ref.iloc[np.repeat(np.arange(N_S), N_L)].values
    quant_rel = (quant.loc[:, Idx[:, :, :]] - stats_ref_val).abs()
    quant_rel["err"] *= 100
    quant["err"] *= 100

    try:
        # utils.to_latex(dirname, quant, table_format)
        utils.to_latex(dirname, quant_rel, table_format)
    except:
        pass


    #quant_rel["err"] *= 100
    #errors.describe().to_csv(os.path.join(dirname, 'errors_describe.csv'))
    #f, axes = plt.subplots(1, 2, figsize=[10., 5.])
    df_reset = quant.reset_index()
    df_plot = pd.melt(df_reset, id_vars='try')
    df_reset_rel = quant_rel.reset_index()
    df_plot_rel = pd.melt(df_reset_rel, id_vars="var")
    rp = sns.relplot(
        #data = df_plot.query('layer > 0'),
        data=df_plot_rel,
        #col='log_mult',
        hue='set',
        hue_order=["train", "test"],
        #dodge=False,
        col='stat',
        col_order=["loss", "err"],
        #col='set',
        #style='layer',
        #col='log_mult',
        x='layer',
        y='value',
        kind='line',
        ci='sd',
        palette=palette,
        #ax=axes[0],
        #kind='line',
        #ylabel='%',
        #ci='sd',
        #col_wrap=2,
        facet_kws={
           'sharey': False,
           'sharex': True
        }
    )
    rp.axes[0,0].set_title("Loss")
    rp.axes[0,0].set_ylabel("absolute delta loss")
    rp.axes[0,1].set_title("Error")
    rp.axes[0,1].set_ylabel("absolute delta error (%)")

    rp.legend.set_title("Datasets")
    # rp.fig.set_size_inches(11,4)
    #rp.axes[0,0].margins(.05)
    #rp.axes[0,1].margins(.05)
    # rp.legend.set_title("Datasets")
    # rp.fig.set_size_inches(12, 4.5)
    # rp.axes[0,0].margins(.05)
    # rp.axes[0,1].margins(.05)
    rp.set(xticks=range(N_L))
    # xlabels=np.arange(N_L)
    # rp.axes[0,0].set_xticklabels(np.arange(N_L))
    # rp.axes[0,1].set_xticklabels(np.arange(N_L))
    #rp.set_xticks(len(xlabels))
    # rp.set_xlabels(xlabels)
    rp.axes[0,0].set_xlabel("layer index l")
    rp.axes[0,1].set_xlabel("layer index l")


    if args_model is not None:
        rp.fig.suptitle("(A) FCN {}".format(args_model.dataset.upper()))


    # try:
        # # vl_left = rp.axes[0,0].viewLim
        # # Dx = vl_left[1][0] - vl_left[0][0]
        # # Dy = vl_left[1][1] - vl_left[0][1]
        # ax0 = rp.axes[0,0]
        # pt = (ax0.viewLim.x0,0)
        # #(0,0) in axes coordinates
        # x,y = (ax0.transData + ax0.transAxes.inverted()).transform(pt)
        # K = 0.1
        # x = x-K
        # rp.axes[0,0].text(x, y+K,   "{:.2f}".format(stats_ref["loss"]["train"]), color=palette[0], transform=rp.axes[0,0].transAxes)
        # rp.axes[0,0].text(x, y-K, "{:.2f}".format(stats_ref["loss"]["test"]), color=palette[1], transform=rp.axes[0,0].transAxes)

        # # vl_right = rp.axes[0,1].viewLim
        # # Dx = vl_right[1][0] - vl_right[0][0]
        # # Dy = vl_right[1][1] - vl_right[0][1]
        # ax1 = rp.axes[0,1]
        # pt = (ax1.viewLim.x0,0)
        # #(0,0) in axes coordinates
        # x,y = (ax1.transData + ax1.transAxes.inverted()).transform(pt)
        # K = 0.1
        # x = x-K
        # rp.axes[0,1].text(x, y+K, "{:.2f}".format(100*stats_ref["err"]["train"]), color=palette[0], transform=rp.axes[0,1].transAxes)
        # rp.axes[0,1].text(x, y-K, "{:.2f}".format(100*stats_ref["err"]["test"]), color=palette[1], transform=rp.axes[0,1].transAxes)
    # except:
        # pass

    sns.lineplot(
        #data=rel_losses.min(axis=0).to_frame(name="loss"),
        data=df_plot_rel.query("stat=='loss'").pivot(index="var", columns=col_order).min(axis=0).to_frame(name="value"),
        #hue="width",
        hue="set",
        hue_order=["train", "test"],
        #col="stat",
        #col_order=["loss", "error"],
        x="layer",
        y="value",
        #kind='line',
        #legend="full",
        #style='set',
        legend=False,
        ax=rp.axes[0,0],
        alpha=0.5,
        #style='layer',
        #markers=['*', '+'],
        dashes=[(2,2),(2,2)],
    )
    for ax in rp.axes[0,0].lines[-2:]:  # the last two
        ax.set_linestyle('--')



    sns.lineplot(
        #data=rel_losses.min(axis=0).to_frame(name="loss"),
        data=df_plot_rel.query("stat=='err'").pivot(index="var", columns=col_order).min(axis=0).to_frame(name="value"),
        #hue="width",
        hue="set",
        hue_order=["train", "test"],
        #col="stat",
        #col_order=["loss", "error"],
        x="layer",
        y="value",
        #kind='line',
        #legend="full",
        #style='set',
        legend=False,
        ax=rp.axes[0,1],
        alpha=0.5,
        #palette=sns.color_palette(n_colors=N_L),
        #style='layer',
        markers=True,
        dashes=[(2,2),(2,2)],
    )
    # rp.axes[0,1].lines[-1].set_linestyle('--')

    for ax in rp.axes[0,1].lines[-2:]:  # the last two
        ax.set_linestyle('--')

    # if stats_ref is not None:
        # sns.lineplot(
            # data=stats_ref.query('stat=="loss"').reset_index(),  # repeat the datasaet N_L times
            # hue='set',
            # hue_order=["train", "test"],
            # ax=rp.axes[0,0],
            # x=np.tile(np.linspace(1, N_L, num=N_L), 2),
            # style='set',
            # dashes=True,
            # legend=False,
            # #y="value",)
            # )


        # sns.lineplot(
            # data=stats_ref.query('stat=="err"').iloc[np.tile(np.arange(2), N_L)].reset_index(),  # repeat the datasaet N_L times
            # hue='set',
            # hue_order=["train", "test"],
            # ax=rp.axes[0,1],
            # x=np.tile(np.linspace(0, N_L, num=N_L), 2),
            # style='set',
            # dashes=True,
            # legend=False,
            # #dashes=[(2,2),(2,2)],
            # #y="value",)
            # )
    # sns.lineplot(
        # #data=df_plot.stats_ref.query('stat=="err"').iloc[np.tile(np.arange(2), N_L)].reset_index(),  # repeat the datasaet N_L times
        # data=df_plot.query('stat=="loss"').pivot(index="var", columns=col_order).min(axis=0).to_frame(name="value"),
        # hue='set',
        # hue_order=["train", "test"],
        # ax=rp.axes[0,0],
        # #x=np.tile(np.linspace(0, N_L, num=N_L), 2),
        # x='layer',
        # alpha=0.5,
        # style='set',
        # dashes=True,
        # legend=False,
        # #dashes=[(2,2),(2,2)],
        # y="value",)

    # sns.lineplot(
        # #data=df_plot.stats_ref.query('stat=="err"').iloc[np.tile(np.arange(2), N_L)].reset_index(),  # repeat the datasaet N_L times
        # data=df_plot.query('stat=="err"').pivot(index="var", columns=col_order).min(axis=0).to_frame(name="value"),
        # hue='set',
        # hue_order=["train", "test"],
        # ax=rp.axes[0,1],
        # #x=np.tile(np.linspace(0, N_L, num=N_L), 2),
        # x='layer',
        # alpha=0.5,
        # style='set',
        # dashes=True,
        # legend=False,
        # #dashes=[(2,2),(2,2)],
        # y="value",)

    #rpset_xticklabels(range(N_L))
    # if is_vgg:
        # xlabels=["conv1", "conv2", "conv3", "conv4", "conv5", "conv6", "conv7", "conv8", "fc1", "fc2"]
        # #mp.set_xticks(len(xlabels))
        # rp.set_xlabels(xlabels)

    #if stats_ref is not None:
    plt.savefig(fname=os.path.join(dirname, 'relplot.pdf'), bbox_inches="tight")




    plt.figure()
    #df_reset = quant.().reset_index()
    #df_plot = pd.melt(df_reset, id_vars='try')
    bp = sns.relplot(
        data=df_plot.pivot(index="var", columns=col_order).min(axis=0).to_frame(name="value"),
        #col='log_mult',
        hue='set',
        hue_order=["train", "test"],
        #dodge=False,
        col='stat',
        col_order=["loss", "err"],
        #col_order=["train", "test", "val"],
        #kcol="set",
        #col='set',
        #style='layer',
        #col='log_mult',
        x='layer',
        y='value',
        kind='line',
        #ci=100,
        #ax=axes[0],
        #kind='line',
        #ylabel='%',
        #ci=100,
        #col_wrap=2,
        facet_kws={
            'sharey': False,
            'sharex': True
        }
    )
    df_ref = df_plot.query('layer==0')
    bp.axes[0,0].set_title("Loss")
    bp.axes[0,0].set_ylabel("loss")

    bp.axes[0,1].set_title("Error")
    bp.axes[0,1].set_ylabel("absolute error (%)")
    #bp.axes[0,0].plot(quant.columns.levels("layer"), quant.loc[1, (0, "loss")], color=red, label='')

    plt.savefig(fname=os.path.join(dirname, 'min_plot.pdf'))

    fig=plt.figure()
    df_reset = quant.notnull().reset_index()
    df_plot = pd.melt(df_reset, id_vars='try')
    g = sns.relplot(
        data = df_plot,
        #col='',
        #hue='set',
        col='stat',
        x='layer',
        y='value',
        kind='line',
        ci=None,
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
        g.fig.suptitle('ds = {}, width = {}, removed = {}, try = {}'.format(args_model.dataset, width, removed, args.ntry))
    g.set(yscale='linear')
    plt.savefig(fname=os.path.join(dirname, 'plot.pdf'))
    g.set(yscale='log')
    plt.savefig(fname=os.path.join(dirname, 'plot_log.pdf'))


    plt.close('all')
    return

def process_checkpoint(checkpoint):
    '''Read and process a previously computed result stored inside a checkpoint (for the copy test)'''

    quant = checkpoint['quant']
    args = checkpoint['args']
    idx = pd.IndexSlice
    process_df(quant, args.path_output)
    return

def read_csv(file_csv):
    '''Read and process a previously computed result stored inside a checkpoint'''

    idx = pd.IndexSlice
    quant = pd.read_csv(file_csv, header=[0,1,2], index_col=0)
    nlevels = quant.columns.nlevels
    layer_idx = quant.columns.names.index("layer")
    if quant.columns.get_level_values(layer_idx).dtype != int:  # set the type to int for the layers
        new_layer_lvl = list(map(int, quant.columns.get_level_values(layer_idx)))
        levels = [quant.columns.get_level_values(i) if i != layer_idx else new_layer_lvl for i in range(nlevels)]
        cols = pd.MultiIndex.from_arrays(levels, names=quant.columns.names)
        quant.columns = cols
    return quant

def process_csv(file_csv):
    '''Read and process a previously computed result stored inside a checkpoint'''

    quant = read_csv(file_csv)

    file_chkpt = os.path.join(os.path.dirname(file_csv), "checkpoint.pth")
    args_model=None
    if os.path.isfile(file_chkpt ):
        chkpt = torch.load(file_chkpt)
        args_model = chkpt["args"]

    dirname = os.path.dirname(file_csv)
    process_df(quant, dirname, args_model=args_model, save=False)
    return


# def eval_test_set(checkpoint, fname, log_fname):
    # '''Eval the model on the test set'''
    # args = checkpoint['args']
    # quant = checkpoint['quant']
    # train_dataset, test_dataset, num_chs = utils.get_dataset(dataset=args.dataset,
                                                        # dataroot=args.dataroot,
                                                                # )

    # train_loader, size_train,\
        # val_loader, size_val,\
        # test_loader, size_test  = utils.get_dataloader( train_dataset, test_dataset, batch_size =args.batch_size, ss_factor=1, size_max=args.size_max, collate_fn=None, pin_memory=True)
    # classifier = utils.parse_archi(log_fname)
    # loss_test, err_test = eval_epoch(model, test_loader)
    # quant.columns.name = add_sets
    # quant.loc[epoch, ('test', 'loss')] = loss_test
    # quant.loc[epoch, ('test', 'err')] = err_test


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
    #tries = np.arange(1, 1+args.ntry)  # the different tries

    names=['set', 'layer', 'stat']
    columns=pd.MultiIndex.from_product([layers, stats], names=names)
    #index = pd.Index(np.arange(1, start_epoch+args.nepochs+1), name='epoch')
    index = pd.Index(np.arange(1, N_T+1), name='steps')
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

        #if not 'set' in quant.columns.names:
            #checkpoint = eval_test_set(checkpoint, file_entry)

        df_bundle = pd.concat([df_bundle, quant], ignore_index=False, axis=1)

        #df_bundle.loc[Idx[:, (idx_entry,'loss')]] = quant.loc[Idx[epoch, ('train', 'loss')]]
        #df_bundle.loc[Idx[:, (idx_entry,'error')]] = quant.loc[Idx[epoch, ('train', 'err')]]
        epochs[idx_entry] = epoch

    df_bundle.sort_index(axis=1, inplace=True)  # sort for quicker access
    return df_bundle, epochs, args



if __name__ == '__main__':

    torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser('Combine the different vars for experiment A/B')
    #parser.add_argument('--dataset', '-dat', default='mnist', type=str, help='dataset')
    #parser.add_argument('--dataroot', '-droot', default='./data/', help='the root for the input data')
    parser.add_argument('--name', default='eval-copy', type=str, help='the name of the experiment')
    #parser.add_argument('--vary_name', nargs='*', default=None, help='the name of the parameter to vary in the name (appended)')
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
    parser.add_argument('--steps', type=int, default=10, help='The number of steps to take')
    parser.add_argument('--table_format', choices=["wide", "long"], default="long")
    parser.set_defaults(cpu=False)
    parser.add_argument('dirs', nargs='*', help='the directory to process')



    args = parser.parse_args()
    table_format = args.table_format

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    #device = torch.device('cpu')


    dtype = torch.float
    num_gpus = torch.cuda.device_count()

    def get_parent(path):
        return os.path.basename(os.path.dirname(path))

    def parse_width(fname):
        width_regexp = re.compile("W-\(\d\+\)")
        found = width_regexp.find(width_regexp, fname)
        return found[1]




    Idx = pd.IndexSlice
    common_dir = os.path.commonpath(args.dirs)
    path_merge = os.path.join(common_dir, 'merge')
    fnames = []
    for dname in args.dirs:
        fnames.extend(glob.glob(os.path.join(dname, "**", "quant.csv"), recursive=True))

    unique_ids = set(list(map(get_parent, fnames)))
    for uid in unique_ids:  # for every f2 etc experiments
        path_output = os.path.join(path_merge, uid)
        df_merge = pd.DataFrame(index=pd.Index([], name="var"))
        df_ref_merge = pd.DataFrame(index=pd.Index([], name="var"))
        os.makedirs(path_output, exist_ok=True)

        for directory in args.dirs:
            vid = int(''.join([c for c in os.path.basename(directory.rstrip('/')) if c.isdigit()]))
            id_fnames = glob.glob(os.path.join(directory, "**", uid, "quant.csv"), recursive=True)


            for fn in id_fnames: # for all variations
                # width = parse_width(fn)

                quant = read_csv(fn)
                col_order = quant.columns.names
                quant_min=quant.min(axis=0).to_frame(name=vid).transpose()
                quant_min.index.name = "var"
                fn_ref = os.path.join(os.path.dirname(fn), "stats_ref.csv")

                if os.path.isfile(fn_ref):
                    quant_ref = pd.read_csv(fn_ref, index_col=[0,1]).transpose().sort_index(axis=1)
                    # levels_ref = list([[width]] +quant_ref.columns.levels)
                    # quant_ref.columns = pd.MultiIndex.from_product(levels_ref,
                                                            # names= ['width'] + quant_ref.columns.names,
                                                            # )
                    quant_ref.index = [vid]
                else:
                    quant_ref = quant.loc[1, Idx[:, :, 0]].dropna().to_frame().transpose().droplevel("layer", axis=1)
                    quant_ref.index = [vid]




                df_merge = pd.concat([df_merge, quant_min], ignore_index=False, axis=0)
                df_ref_merge = pd.concat([df_ref_merge, quant_ref], ignore_index=False, axis=0)

        df_merge.sort_index(axis=1, inplace=True)
        df_ref_merge.sort_index(axis=1, inplace=True)
        df_merge.to_csv(os.path.join(path_output, 'min.csv'))
        df_ref_merge.to_csv(os.path.join(path_output, 'ref.csv'))
