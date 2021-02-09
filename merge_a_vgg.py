import torch
import numpy as np
import pandas as pd
import os
import sys
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


import models

import torch.optim
import torch
import argparse
import utils


#from torchvision import models, datasets, transforms

try:
    from tqdm import tqdm
except:
    def tqdm(x): return x



def process_df(quant, dirname, stats_ref=None, args=None, args_model=None, save=True):

    global table_format

    idx = pd.IndexSlice
    #losses = quant.loc[:, idx[:, '#loss']]
    #errors = quant.loc[:, idx[:, 'error']]

    col_order = ["stat", "set", "layer"]
    if quant.columns.names != col_order:
        # the order is
        # perform pivot
        quant = pd.melt(quant.reset_index(), id_vars="draw").pivot(index="draw", columns=col_order, values="value")


    if stats_ref is not None:
        if stats_ref.index.names != ["stat", "set"]:
            stats_ref = stats_ref.reorder_levels(["stat", "set"]).sort_index(axis=0)

    quant.sort_index(axis=1, inplace=True)


    if save:
        quant.to_csv(os.path.join(dirname, 'quant.csv'))
        stats_ref.to_csv(os.path.join(dirname, 'stats_ref.csv'))
        quant.groupby(level=["stat", "set"], axis=1).describe().to_csv(os.path.join(dirname, 'describe.csv'))

    # if len(stats_ref.keys()==1):
        # stats_ref = stats_ref[stats_ref.keys()[0]]
    N_L = len(quant.columns.unique(level="layer")) # number of hidden layers
    #N_sets = len(quant.columns.unique(level="set"))
    N_sets = 2 # only train and test



    palette=sns.color_palette(n_colors=N_sets)

    df_reset = quant.reset_index()
    N_S = len(stats_ref)
    stats_ref_val = stats_ref.iloc[np.repeat(np.arange(N_S), N_L)].transpose().values
    quant_rel = (quant - stats_ref_val).abs()
    quant_rel["error"] *= 100
    quant["error"] *= 100

    # else:
        # table = quant_describe[["mean", "std", "min"]]

    # formaters =
    try:
        # utils.to_latex(dirname, quant, table_format, is_vgg=True)
        utils.to_latex(dirname, quant_rel, table_format, is_vgg=True)
    except:
        pass
    df_plot = pd.melt(df_reset, id_vars='draw')

    df_reset_rel = quant_rel.reset_index()
    df_plot_rel = pd.melt(df_reset_rel, id_vars="draw")

    rp = sns.relplot(
        data=df_plot_rel.pivot(index="draw", columns=col_order).min(axis=0).to_frame(name="value"),
        #col='log_mult',
        hue='set',
        hue_order=["train", "test"],
        #dodge=False,
        col='stat',
        col_order=["loss", "error"],
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
        #ci=100,
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
    # rp.fig.set_size_inches(11, 4)
    #rp.axes[0,0].margins(.05)
    #rp.axes[0,1].margins(.05)
    xlabels=["0", "conv1", "conv2", "conv3", "conv4", "conv5", "conv6", "conv7", "conv8", "fc1", "fc2"]
    rp.set(xticks=range(0, len(xlabels)))
    # rp.set_xticklabels(xlabels)
    # rp.axes[0,0].locator_params(axis='x', nbins=len(xlabels))
    # rp.axes[0,1].locator_params(axis='x', nbins=len(xlabels))

    rp.set_xticklabels(xlabels, rotation=30)
    # rp.axes[0,0].set_xticklabels(xlabels, rotation=30)
    # rp.axes[0,1].set_xticklabels(xlabels, rotation=30)
    #rp.set_xticks(len(xlabels))
    #rp.set_xlabels(xlabels)


    if args_model is not None:
        rp.fig.suptitle("(A) VGG {}".format(args_model.dataset.upper()))

    plt.savefig(fname=os.path.join(dirname, 'relplot.pdf'), bbox_inches="tight")

    plt.figure()
    rp = sns.relplot(
        data=df_plot.pivot(index="draw", columns=col_order).min(axis=0).to_frame(name="value"),
        hue='set',
        hue_order=["train", "test"],
        col='stat',
        col_order=["loss", "error"],
        x='layer',
        y='value',
        kind='line',
        facet_kws={
            'sharey': False,
            'sharex': True
        }
    )
    df_ref = df_plot.query('layer==0')
    rp.axes[0,0].set_title("Loss")
    rp.axes[0,0].set_ylabel("loss")

    rp.axes[0,1].set_title("Error")
    rp.axes[0,1].set_ylabel("error (%)")

    plt.savefig(fname=os.path.join(dirname, 'rel_plot.pdf'))

    fig=plt.figure()
    df_reset = quant.notnull().reset_index()
    df_plot = pd.melt(df_reset, id_vars='draw')
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
        g.fig.suptitle('ds = {}, width = {}, removed = {}, draw = {}'.format(args_model.dataset, width, removed, args.ndraw))
    g.set(yscale='linear')
    plt.savefig(fname=os.path.join(dirname, 'plot.pdf'))
    g.set(yscale='log')
    plt.savefig(fname=os.path.join(dirname, 'plot_log.pdf'))


    plt.close('all')
    return


def process_csv(file_csv):
    '''Read and process a previously computed result stored inside a checkpoint'''

    global device
    idx = pd.IndexSlice
    quant = pd.read_csv(file_csv, header=[0,1,2], index_col=0)
    file_ref = os.path.join(os.path.dirname(file_csv), "stats_ref.csv")
    if os.path.isfile(file_ref):
        stats_ref = pd.read_csv(file_ref, index_col=[0,1])
    layer_idx = quant.columns.names.index("layer")
    if quant.columns.get_level_values(layer_idx).dtype != int:  # 0 are the layers
        new_layer = [int(c) for c in quant.columns.levels[layer_idx]]
        new_layer.sort()
        levels = list(quant.columns.levels[:layer_idx] + [new_layer] + quant.columns.levels[layer_idx+1:])
        cols = pd.MultiIndex.from_product(levels, names=quant.columns.names)
        quant.columns = cols

    try:
        chkpt_model = torch.load(os.path.join(os.path.dirname(os.path.dirname(file_csv)), 'checkpoint.pth'), map_location=device)
        args_model = chkpt_model['args']
    except:
        args_model =None

    #quant.loc[:, idx[:, :, 'error']] *= 100  # in percent

    dirname = os.path.dirname(file_csv)
    process_df(quant, dirname, stats_ref, args_model=args_model, save=False)
    return




if __name__ == '__main__':

    torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser('Evaluating a copy of a classifier with removed units')
    parser_device = parser.add_mutually_exclusive_group()
    parser_device.add_argument('--cpu', action='store_true', dest='cpu', help='force the cpu model')
    parser_device.add_argument('--cuda', action='store_false', dest='cpu')
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


    Idx = pd.IndexSlice
    for directory in args.dirs:
        # directory is the root of the model
        # e.g. root/fraction-2/entry_10
        if os.path.isfile(directory) and directory.endswith('.csv'):
            process_csv(directory)
            sys.exit(0)



        models = glob.glob(os.path.join(os.path.dirname(directory.rstrip(os.sep)), "checkpoint.pth"))


        for f in models:
            model = torch.load(f, map_location=device)
            quant_model = model["quant"].dropna()
            args_model = model["args"]
            idx_min = quant_model.idxmin(axis=0)["train", "loss"] # the epochs for each draw
            stats_ref =  quant_model.loc[idx_min]

            d_m = os.path.dirname(f)  # the directory
            entry_dirs = glob.glob(os.path.join(d_m, "**", "entry_*"), recursive=True) # all the entries
            roots = set(list(map(os.path.dirname, entry_dirs)))  # the names of the



            # root is e.g. fraction-2, and will be the root of the figure
            for root in roots:
                df_merge = pd.DataFrame()
                df_min = pd.DataFrame()  # only take the minimum over the epochs
                entries = glob.glob(os.path.join(root, "entry_*"), recursive=False)
                entries_id = map(os.path.basename, entries)
                entries_id = [int(x.split('_')[1]) for x in  entries_id]
                entries_id.sort()
                for eid in entries_id:
                    entry_dir = os.path.join(root, f"entry_{eid}")
                    f_checkpoints  = glob.glob(os.path.join(entry_dir, "checkpoint_draw_*.pth"), recursive=False)
                    chkpt_id = map(os.path.basename, f_checkpoints)
                    chkpt_id = [int(x.split('.')[0].split('_')[-1]) for x in  chkpt_id]
                    chkpt_id.sort(reverse=True)
                    if len(chkpt_id) == 0:
                        continue
                    id_max = chkpt_id[0]
                    f =os.path.join(entry_dir, f"checkpoint_draw_{id_max}.pth")

                    f_min =os.path.join(entry_dir, f"checkpoint_min_draw_{id_max}.pth")

                    #for f in f_checkpoints:
                    name = os.path.basename(f).split('.')[0]

                    #idx_draw = int(name.split('_')[-1])
                    chkpt = torch.load(f, map_location=device)
                    quant = chkpt['quant'].sort_index(axis=1).dropna()#.min(axis=0)
                    if quant.empty:  # empty dataframe
                        continue
                    # sort by train loss
                    idx_min = quant.idxmin(axis=0)["train", "loss"].to_frame(name="epoch") # the epochs for each draw
                    #draw_range = idx_min.dropna().index
                    #levels = list( quant.columns.levels[:2] + [[eid]] + quant.columns.levels[-1:])
                    #data=quant.pivot(index="epochs", columns=).min(axis=0).to_frame(name="value"),
                    #col_order =["set", "stat", "layer", "draw"]
            #quant = pd.melt(quant.reset_index(), id_vars="epochs").pivot(index="epochs", columns=col_order, values="value")

                    #quant.columns = pd.MultiIndex.from_product(levels,
                    #                                           names=  quant.columns.names[:2] + ['layer', 'draw'],

#                                                        )

                    #idx_min = pd.melt(idx_min.reset_index(), id_vars="epochs").pivot(index="epochs", columns=['layer', 'draw'], values="value")
                    levels = list( [[eid]] + quant.columns.levels[:2])
                    names = ['layer'] + quant.columns.names[:2]

                    columns_min = pd.MultiIndex.from_product(list(quant.columns.levels[:2]), names=quant.columns.names[:2])
                    n_draws = chkpt["draws"]
                    index = pd.Index(np.arange(1, n_draws+1), name="draw")
                    quant_min =  pd.DataFrame(index=index, columns = columns_min, dtype=float)

                    for d in index:  # select the min over the epochs
                        quant_min.loc[d] = quant.loc[idx_min.loc[d, :], Idx[:, :, d]].values
                    quant_min.columns = pd.MultiIndex.from_product(levels, names = names)

                    min_epoch = None

                    # quant.columns = pd.MultiIndex.from_arrays([quant.columns.get_level_values(1), quant.columns.get_level_values(0), level_width],
                                                            # quant.columns.names[::-1] + ['width'],
                                                            # )

                    df_merge = pd.concat([df_merge, quant_min], ignore_index=False, axis=1)

                df_merge.sort_index(axis=1, inplace=True)
                #df_merge[:, Idx["error", :, :]] *= 100  # in percents
                process_df(df_merge, root, stats_ref, args_model=args_model)



