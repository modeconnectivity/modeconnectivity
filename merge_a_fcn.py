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
        quant = pd.melt(quant.reset_index(), id_vars="draw").pivot(index="draw", columns=col_order, values="value")
    idx_order = ["stat", "set"]
    if stats_ref.index.names !=idx_order:
        stats_ref = stats_ref.reorder_levels(idx_order).sort_index(axis=0)

    quant_describe = quant.groupby(level=["stat", "set"], axis=1, group_keys=False).describe()
    if save:
        quant.to_csv(os.path.join(dirname, 'quant.csv'))
        if stats_ref is not None:
            stats_ref.to_csv(os.path.join(dirname, 'stats_ref.csv'))
        quant_describe.to_csv(os.path.join(dirname, 'describe.csv'))



    # table_err_train = table["error"]["train"]
    #quant.loc[:, Idx[:, :, "error"]] *= 100
    if len(stats_ref.keys()) == 1:
        stats_ref = stats_ref[stats_ref.keys()[0]]
    #quant["error"] *= 100
    #stats_ref_copy  = stats_ref.copy()
    #stats_ref_copy["error"] = stats_ref["error"] * 100
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
    quant_rel["error"] *= 100
    quant["error"] *= 100

    try:
        # utils.to_latex(dirname, quant, table_format)
        utils.to_latex(dirname, quant_rel, table_format)
    except:
        pass


    #quant_rel["error"] *= 100
    #errors.describe().to_csv(os.path.join(dirname, 'errors_describe.csv'))
    #f, axes = plt.subplots(1, 2, figsize=[10., 5.])
    df_reset = quant.reset_index()
    df_plot = pd.melt(df_reset, id_vars='draw')
    df_reset_rel = quant_rel.reset_index()
    df_plot_rel = pd.melt(df_reset_rel, id_vars="draw")
    rp = sns.relplot(
        #data = df_plot.query('layer > 0'),
        data=df_plot_rel,
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

    sns.lineplot(
        #data=rel_losses.min(axis=0).to_frame(name="loss"),
        data=df_plot_rel.query("stat=='loss'").pivot(index="draw", columns=col_order).min(axis=0).to_frame(name="value"),
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
        data=df_plot_rel.query("stat=='error'").pivot(index="draw", columns=col_order).min(axis=0).to_frame(name="value"),
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
    #if stats_ref is not None:
    plt.savefig(fname=os.path.join(dirname, 'relplot.pdf'), bbox_inches="tight")




    plt.figure()
    #df_reset = quant.().reset_index()
    #df_plot = pd.melt(df_reset, id_vars='draw')
    bp = sns.relplot(
        data=df_plot.pivot(index="draw", columns=col_order).min(axis=0).to_frame(name="value"),
        #col='log_mult',
        hue='set',
        hue_order=["train", "test"],
        #dodge=False,
        col='stat',
        col_order=["loss", "error"],
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

def process_checkpoint(checkpoint):
    '''Read and process a previously computed result stored inside a checkpoint (for the copy test)'''

    quant = checkpoint['quant']
    args = checkpoint['args']
    idx = pd.IndexSlice
    process_df(quant, args.path_output)
    return

def process_csv(file_csv):
    '''Read and process a previously computed result stored inside a checkpoint'''

    idx = pd.IndexSlice
    quant = pd.read_csv(file_csv, header=[0,1,2], index_col=0)
    if quant.columns.get_level_values(0).dtype != int:  # 0 are the layers
        new_layer = [int(c) for c in quant.columns.levels[0]]
        levels = list([new_layer] + quant.columns.levels[1:])
        cols = pd.MultiIndex.from_product(levels, names=quant.columns.names)
        quant.columns = cols


    file_chkpt = os.path.join(os.path.dirname(file_csv), "checkpoint.pth")
    args_model=None
    if os.path.isfile(file_chkpt ):
        chkpt = torch.load(file_chkpt)
        args_model = chkpt["args"]

    dirname = os.path.dirname(file_csv)
    process_df(quant, dirname, args_model=args_model, save=False)
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



        # entry_dirs = glob.glob(os.path.join(directory, "**", "entry_*"), recursive=True) # all the entries
        # roots = set(list(map(os.path.dirname, entry_dirs)))  # the names of the
        # # root is e.g. fraction-2, and will be the root of the figure
        # for root in roots:
        f_model = None
        #f_checkpoints  = glob.glob(os.path.join(directory, "**", "checkpoint_entry_*.pth"), recursive=True)
        models = glob.glob(os.path.join(directory, "**", "checkpoint.pth"), recursive=True)
        #entries = glob.glob(os.path.join(root, "entry_*"), recursive=False)

        for f in models:
            model = torch.load(f, map_location=device)
            quant_model = model["quant"].dropna()
            args_model = model["args"]
            idx_min = quant_model.idxmin(axis=0)["train", "loss"] # the index of minimum train loss
            stats_ref =  quant_model.loc[idx_min].to_frame()

            d_m = os.path.dirname(f)  # the directory
            entry_files = glob.glob(os.path.join(d_m, "**", "checkpoint_entry_*"), recursive=True) # all the subnetworks files
            exps = set(map(os.path.dirname, entry_files))  # the names of the experiments, to group them

            for dname in exps:  # for aan experiment

                files = glob.glob(os.path.join(dname, "checkpoint_entry_*.pth"), recursive=False)

                df_merge = pd.DataFrame()

                for f in files:

                    chkpt = torch.load(f, map_location=device)

                    quant = chkpt['quant'].sort_index(axis=1).dropna()#.min(axis=0)
                    if quant.empty:  # empty dataframe
                        continue
                    # sort by train loss
                    idx_min = quant.idxmin(axis=0)["train", "loss"].to_frame(name="epoch") # the epochs for each draw
                    steps_range = idx_min.dropna().index
                    #quant.columns = None
                    #quant.columns = pd.MultiIndex.from_product(levels, names = names)

                        #levels = list( quant.columns.levels[:2] + [[eid]] + quant.columns.levels[-1:])
                    #idx_min = pd.melt(idx_min.reset_index(), id_vars="epochs").pivot(index="epochs", columns=['layer', 'draw'], values="value")
                    eid = chkpt['args'].entry_layer
                    levels = list( [[eid]] + quant.columns.levels[:2])
                    names = ['layer'] + quant.columns.names[:2]

                    columns_min = pd.MultiIndex.from_product(list(quant.columns.levels[:2]), names=quant.columns.names[:2])
                    #n_stepss = quant.levels("try")
                    index = quant.columns.levels[-1]#pd.Index(np.arange(1, n_steps+1), name="draw")
                    quant_min =  pd.DataFrame(index=index, columns = columns_min, dtype=float)

                    for d in index:  # select the min over the epochs
                        quant_min.loc[d] = quant.loc[idx_min.loc[d, :], Idx[:, :, d]].values
                    quant_min.columns = pd.MultiIndex.from_product(levels, names = names)

                    df_merge = pd.concat([df_merge, quant_min], ignore_index=False, axis=1)
                min_epoch = None


                df_merge.sort_index(axis=1, inplace=True)
                try:
                    process_df(df_merge, dname, stats_ref, args_model=args_model)
                except RuntimeError as e:
                    print(f"error {e} processing {dname}")

