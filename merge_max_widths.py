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


def check_columns(*args):
    for df in args:
        if df.empty:
            continue
        int_idx_lst = [] # the list for int fields
        if "layer" in df.columns.names:
            int_idx_lst += [df.columns.names.index("layer")]
        if "width" in df.columns.names:
            int_idx_lst += [df.columns.names.index("width")]
        stat_idx = df.columns.names.index("stat")
        nlevels = df.columns.nlevels
        # stat_idx = df.columns.names.index("stat")
# dirname = os.path.dirname(file_csv)
        for idx in int_idx_lst:  # parse to int
            if df.columns.get_level_values(idx).dtype != int:  # 0 are the layers
                new_lvl = list(map(int, df.columns.get_level_values(idx)))
                levels = [df.columns.get_level_values(i) if i != idx else new_lvl for i in range(nlevels)]
                cols = pd.MultiIndex.from_arrays(levels, names=df.columns.names)
                df.columns = cols
        if "err" in df.columns.get_level_values("stat"):
            new_stat_lvl = [s.replace("err", "error") for s in df.columns.get_level_values(stat_idx)]
            # new_stat.sort()
            levels = [df.columns.get_level_values(i) if i != stat_idx else new_stat_lvl for i in range(nlevels)]
            cols = pd.MultiIndex.from_arrays(levels, names=df.columns.names)
            df.columns = cols
        df.index.name = "var"


    return args

def process_df(quant, dirname, args=None, args_model=None, save=True, quant_ref=None):

    global table_format
    Idx = pd.IndexSlice
    check_columns(quant, quant_ref)

    # if (quant_ref is None or quant_ref.empty) and 0 in quant.columns.get_level_values("layer"):
        # quant_ref = quant.loc[:, Idx[:, :, 0, :]].droplevel("layer", axis=1)  # for all the  widths and all the vars

    has_ref = quant_ref is not None and not quant_ref.empty

    if quant.columns.names != ["stat", "set", "width"]:
        # the order is
        # perform pivot
        quant = pd.melt(quant.reset_index(), id_vars="var").pivot(index="var", columns=["stat", "set", "width"], values="value")

    # output_root = os.path.join(dirname, f"merge", uid) if len(uid) > 0 else dirname
    output_root = dirname
    os.makedirs(output_root, exist_ok=True)
    idx = pd.IndexSlice
    cols_error = idx['error',  :, :]
    # N_L = len(quant.columns.unique(level="layer"))  # number of hidden layers
    errors = quant["error"]
    losses = quant["loss"]

    if save:
        quant.to_csv(os.path.join(output_root, 'min.csv'))


    quant.sort_index(axis=1, inplace=True)
    quant.loc[:, cols_error] *= 100  # in %
    if has_ref:
        quant_ref.loc[:, Idx[:, "error", :]] *= 100  # in %
        # quant_ref = stats_ref.agg(['mean', 'count', 'std'])
        # quant_ref.loc['se'] = quant_ref.loc['std'] / np.sqrt(quant_ref.loc['count'])
        # quant_ref.loc['ci95'] = 1.96 * quant_ref.loc['se']  # 95% CI
    quant.groupby(level=["stat", "set"], axis=1, group_keys=False).describe().to_csv(os.path.join(output_root, 'describe.csv'))
    #csvlosses.to_csv(os.path.join(output_root, 'losses.csv'))
    #errors.to_csv(os.path.join(output_root, 'errors.csv'))

    #quant_describe = pd.DataFrame(group.describe().rename(columns={'value': name}).squeeze()
    #                              for (name, group) in quant.groupby(level=["stat", "set"], axis=1))
    #quant_describe.to_csv(os.path.join(output_root, 'describe.csv'))

    #fig=plt.figure()
    #f, axes = plt.subplots(1, 2, figsize=[10., 5.])
    if args is not None and args.xlim is not None:
        idx_width = quant.columns.names.index("width")
        if len(args.xlim)==2:
            quant = quant.loc[:, args.xlim[0] <= quant.columns.get_level_values(idx_width) <= args.xlim[1]]
        elif len(args.xlim)==1:
            quant = quant.loc[:, quant.columns.get_level_values(idx_width) <= args.xlim[0]]



    # df_reset = quant.reset_index()
    # df_plot = pd.melt(df_reset, id_vars='var')#.query("layer>0")
    # df_plot_no_0 = df_plot.query('layer>0')
    # df_plot_0 = df_plot.query('layer==0')
    #relative quantities
    # N_S = len(quant_ref.columns)  # should be the number of stats
    # quant_ref_val = quant_ref.iloc[np.repeat(np.arange(N_S), N_L)].values
    # quant_rel = (quant.loc[:, Idx[:, :, 1:]] - quant_ref_val).abs()
    #quant_plus = quant.loc[:, Idx[:, :, 1:]] + quant_ref + 1e-10
    #quant_rel /= quant_plus
    #quant_rel *= 2

    # utils.to_latex(output_root, quant.loc[:, Idx[:, :, 1:]], table_format, key_err="error")
    # utils.to_latex(output_root, quant_rel, table_format, key_err="error")

    # df_reset_rel = quant_rel.reset_index()
    # df_plot_rel = pd.melt(df_reset_rel, id_vars="run")

    # palette=sns.color_palette(n_colors=2)  # the two datasets
    N_L=1
    if 'ds' in os.path.basename(output_root):
        # experiment B, orange
        palette = sns.color_palette(n_colors=1)[0:1]
    else:
        palette = sns.color_palette(n_colors=2)[1:2]

    palette_ref = sns.color_palette(n_colors=3)[2:3]
    # if N_L >= 3:
        # pt = sns.color_palette(n_colors=N_L+1)  # remove green
        # palette = pt[0:2] + pt[3:]
        # palette_ref = pt[2:3]
    # else:
        # palette=sns.color_palette(n_colors=N_L)  # the N_L layers
        # palette_ref = sns.color_palette(n_colors=3)[2:3]

    is_vgg=False
    # dataset="MNIST"
    logstr = "_log" if args.yscale == "log" else ""
    labels = ["max", "ref."]#[str(i) for i in quant.columns.get_level_values("layer").unique()] + has_ref*["Ref."]
    # fig.suptitle("{} {}".format('VGG' if is_vgg else 'FCN', dataset.upper()))
    k = 0

    #rp.set_axis_labels("layer", "Loss", labelpad=10)
    #quant.loc[1, Idx["loss", :, 0]].lineplot(x="layer_ids", y="value", hue="")
    for i, stat in enumerate(["loss","error" ]):
        for j, setn in enumerate(["train", "test"]):
            if stat == "loss" and setn=="test":
                continue
            if stat == "error" and setn=="train":
                continue
            # axes[k] = rp.axes[j,i]
            # ax = axes[k]
            plt.figure()
            fig,ax = plt.subplots(1,1,figsize=(4,4))

            # df_plot = quant.loc[:, Idx[:, stat, setn, :]].min(axis=0).to_frame(name="value")
            df_plot= pd.melt(quant.loc[:, Idx[stat, setn, :, :]].reset_index(), id_vars="var")#.min(axis=0).to_frame(name="value")
            lp = sns.lineplot(
                #data=rel_losses.min(axis=0).to_frame(name="loss"),
                # data=df_plot_rel if not is_vgg else df_plot_rel.pivot(index="var", columns=col_order).min(axis=0).to_frame(name="value"),
                data=df_plot,
                #hue="width",
                # hue="layer",
                # hue_order=["A", "B"],
                x="width",
                y="value",
                legend=None,
                # style='set',
                ci='sd',
                palette=palette,
                #style='layer',
                markers=False,
                ax=ax,
                dashes=True,
                #legend_out=True,
                #y="value",
            )
            # widths = quant.columns.get_level_values("width").unique()
            # b = widths[-1]
            # p = int(math.log(b, 10))
            # k = int(math.floor(b / (10**(math.floor(math.log(b, 10))))))

            # xticks=[widths[0]] + [i * 10**p for i in range(1,k)] + [widths[-1]]
            # lp.set(xticks=xticks)
            # lp.set(xticks=range(0, len(xlabels)))
            # rp.set_xticklabels(xlabels)
            # rp.axes[0,0].locator_params(axis='x', nbins=len(xlabels))
            # rp.axes[0,1].locator_params(axis='x', nbins=len(xlabels))

            # lp.set_xticklabels(xlabels, rotation=30*(is_vgg))
            # if not split:
            ax.set_title("{} {}{}".format(setn.title()+(setn=="train")*"ing", stat.title(), " (%)" if stat=="error" else ''))
            # ylabel = stat if stat == "loss" else "error (%)"


            # ax.set_xlabel("width")
            ax.tick_params(labelbottom=True)
            if has_ref:
                df_ref = quant_ref.loc[:, Idx[:, stat,setn]]

                sns.lineplot(data=pd.melt(df_ref.reset_index(), id_vars="var"),
                            ax=ax,
                            hue='stat',
                            # hue_order=["train", "test"],
                            # alpha=0.5,
                            x="width",
                            y="value",
                            palette=palette_ref,
                            # hue=["
                            legend=False,
                            )
                # ax.plot(df_ref, c='g', ls=':')
                ax.set_ylabel(None)

                for l in ax.lines[-1:]:
                    l.set_linestyle(':')
                # l.set_color('g')
            ax.set_yscale(args.yscale)

            if k == 1:
                fig.legend(handles=ax.lines, labels=labels, title="Layer", bbox_to_anchor=(0.9,0.8), borderaxespad=0.)
            k+=1

            plt.savefig(fname=os.path.join(output_root, f"{setn}_{stat}{logstr}.pdf"), bbox_inches='tight')
    # fig.subplots_adjust(top=0.85)
    # fig.legend(ax.lines, labels=["A", "B", "Reference"], title="Experiment", loc="center right")

    # palette=sns.color_palette(n_colors=2)  # the two experiments
    fig, axes = plt.subplots(1, 1, figsize=(4, 4))

    # xlabels=[str(i) for i in range(N_W)]
    is_vgg=False
    # dataset="MNIST"
    # fig.suptitle("{} {}".format('VGG' if is_vgg else 'FCN', dataset.upper()))
    k = 0

    #rp.set_axis_labels("layer", "Loss", labelpad=10)
    #quant.loc[1, Idx["loss", :, 0]].lineplot(x="layer_ids", y="value", hue="")
    for i, stat in enumerate(["error"]):
        for j, setn in enumerate(["train"]):#, "test"]):
            if stat == "loss" and setn=="test":
                continue
            # axes[k] = rp.axes[j,i]
            # ax = axes[k]
            ax = axes

            # df_plot = quant.loc[:, Idx[:, stat, setn, :]].min(axis=0).to_frame(name="value")
            df_plot= pd.melt(quant.loc[:, Idx[stat, setn, :, :]].reset_index(), id_vars="var")
            # df_plot= quant.loc[:, Idx[stat, setn, 1:, :]].min(axis=0).to_frame(name="value")
            lp = sns.lineplot(
                #data=rel_losses.min(axis=0).to_frame(name="loss"),
                # data=df_plot_rel if not is_vgg else df_plot_rel.pivot(index="var", columns=col_order).min(axis=0).to_frame(name="value"),
                data=df_plot,
                #hue="width",
                # hue="layer",
                # hue_order=["A", "B"],
                x="width",
                y="value",
                legend=None,
                # style='set',
                ci='sd',
                palette=palette,
                #style='layer',
                markers=False,
                ax=ax,
                dashes=True,
                #legend_out=True,
                #y="value",
            )
            widths = quant.columns.get_level_values("width").unique()
            b = widths[-1]
            p = int(math.log(b, 10))
            k = int(math.floor(b / (10**(math.floor(math.log(b, 10))))))

            xticks=[widths[0]] + [i * 10**p for i in range(1,k)] + [widths[-1]]
            lp.set(xticks=xticks)
            # lp.set_xticklabels(xticks)
            # rp.axes[0,0].locator_params(axis='x', nbins=len(xlabels))
            # rp.axes[0,1].locator_params(axis='x', nbins=len(xlabels))

            # lp.set_xticklabels(xlabels, rotation=30*(is_vgg))
            ax.set_title("{} {}{}".format(setn.title(), stat.title(), " (%)" if stat=="error" else ''))
            # ylabel = stat if stat == "loss" else "error (%)"


            ax.set_xlabel("width")
            if has_ref:
                df_ref = quant_ref.loc[:, Idx[:, stat,setn]]

                sns.lineplot(data=pd.melt(df_ref.reset_index(), id_vars="var"),
                            ax=ax,
                            hue='stat',
                            # hue_order=["train", "test"],
                            # alpha=0.5,
                            x="width",
                            y="value",
                            palette=palette_ref,
                            legend=False,
                            )
                # ax.plot(df_ref, c='g', ls=':')
                ax.set_ylabel(None)
                ax.set_yscale(args.yscale)

                for l in ax.lines[-1:]:
                    l.set_linestyle(':')
                # l.set_color('g')

            k+=1

    # fig.subplots_adjust(top=0.85)
    fig.legend(handles=ax.lines, labels=labels, title="Layer", bbox_to_anchor=(0.9,0.8), borderaxespad=0.)#, bbox_transform=fig.transFigure)
    # fig.legend(ax.lines, labels=[], title="Experiment", loc="center right")
    plt.margins()
    plt.savefig(fname=os.path.join(output_root, "error_train.pdf"), bbox_inches='tight')

    # for i in range(N_L):

    # or i, stat in enumerate(["loss", "error"]):
        # for j, setn in enumerate(["train", "test"]):
            # if stat == "loss" and setn=="test":
                # continue
            # axes[k] = rp.axes[j,i]

            # ax.plot((0,quant_ref[stat, i][0]), slope=0, ls=":", zorder=2, c='g')

            # sns.lineplot(
                # #data=rel_losses.min(axis=0).to_frame(name="loss"),
                # data=df_plot_rel.query(f"stat=='{stat}' & layer=={i+1}").pivot(index="var", columns=cols).min(axis=0).to_frame(name="value"),
                # #hue="width",
                # hue="set",
                # hue_order=["train", "test"],
                # #col="stat",
                # #col_order=["loss", "error"],
                # x="width",
                # y="value",
                # #kind='line',
                # #legend="full",
                # # style='layer',
                # legend=False,#'brief',
                # ax=ax_loss,
                # alpha=0.5,
                # #style='layer',
                # #markers=['*', '+'],
                # # dashes=[(2,2),(2,2)],
                # # legend_out=True,
            # )
            # mp.axes[0,1].set_ylabel("error (%)")
            # mp.axes[0,0].set_title("Train Error")
            # mp.axes[0,1].set_title("Test Error")

            # ax_err = mp.axes[i, 1] if N_L == 1 else mp.axes[1, i]
            # ax_err.set_title(f"Error" + (N_L>1)* f", Layer {i+1}")
            # ax_err.set_ylabel("absolute delta error (%)")
            # # mp.axes[1,1].set_title("Test Loss")

        # for ax in ax_loss.lines[-2:]:  # the last two
            # ax.set_linestyle('--')
        # leg_loss = mp_loss.get_legend()



        # sns.lineplot(
            # #data=rel_losses.min(axis=0).to_frame(name="loss"),
            # data=df_plot_rel.query(f"stat=='error' & layer=={i+1}").pivot(index="var", columns=cols).min(axis=0).to_frame(name="value"),
            # #hue="width",
            # hue="set",
            # hue_order=["train", "test"],
            # #col="stat",
            # #col_order=["loss", "error"],
            # x="width",
            # y="value",
            # #kind='line',
            # #legend="full",
            # # style='layer',
            # # legend='brief',
            # legend=False,
            # ax=ax_err,
            # alpha=0.5,
            # #palette=sns.color_palette(n_colors=N_L),
            # #style='layer',
            # markers=True,
            # # dashes=[(2,2),(2,2)],
        # )
        # # rp.axes[0,1].lines[-1].set_linestyle('--')

        # for ax in ax_err.lines[-2:]:  # the last two + legend
            # ax.set_linestyle('--')

    # mp.add_legend(plt.legend(mp.axes[0,1].lines[-2:], ("min",)))
    # mp_err.legend().set_title("min")
    # # mp.axes[1,1].set_ylabel("loss")
    plt.margins()
    plt.savefig(fname=os.path.join(output_root,f"min_quant_{stat}.pdf"), bbox_inches="tight")

    # plt.figure()

    # lp = sns.lineplot(
        # data=rel_error.min(axis=0).to_frame(name="error"),
        # hue="layer",
        # x="width",
        # y="error",
        # legend="full",
        # palette=sns.color_palette(n_colors=N_L),
        # style='layer',
        # markers=True,
        # #y="value",
    # )
    # lp.axes.set_ylabel("relative difference")
    # lp.axes.set_title("Error")
    # rp.axes[0,1].set_ylabel("error")
    # plt.savefig(fname=os.path.join(output_root, "rel_error.pdf"))



    # only_min = select_min(quant)  # select the draw with minimum error
    # only_min_plot = pd.melt(only_min.reset_index(), id_vars='step')



    # m = sns.relplot(data=only_min_plot.query('layer > 0'),
    # #m = only_min_plot.plot(x='layer', kind='line', y='value')
        # col='stat',
        # x='layer',
        # y='value',
        # kind='scatter',
        # facet_kws={
            # 'sharey': False,
            # 'sharex': True
        # }
    # )
    # plt.savefig(fname=os.path.join(output_root, 'plot_min.pdf'))
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
    check_columns(quant)
    return quant

def process_csv(file_csv, args=None):
    '''Read and process a previously computed result stored inside a checkpoint'''

    idx = pd.IndexSlice
    quant = pd.read_csv(file_csv, header=[0,1,2,3], index_col=0)
    file_ref = os.path.join(os.path.dirname(file_csv), "ref.csv")
    quant_ref = pd.read_csv(file_ref, header=[0,1,2], index_col=0)
    if args.vlim is not None:
        quant = quant.loc[1:args.vlim, :]
    # if quant.columns.get_level_values(width_idx).dtype != int:  # 0 are the layers
        # new_layer_lvl = list(map(int, quant.columns.get_level_values(width_idx)))
        # levels = [quant.columns.get_level_values(i) if i != layer_idx else new_layer_lvl for i in range(nlevels)]
        # cols = pd.MultiIndex.from_arrays(levels, names=quant.columns.names)
        # quant.columns = cols
    # uid = ''
    process_df(quant, os.path.dirname(file_csv),  args, save=False, quant_ref=quant_ref)
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
        # test_loader, size_test  = utils.get_dataloader( train_dataset, test_dataset, batch_size =args.batch_size, ss_factor=1, size_max=args.size_max, collate_fn=None, pin_memory=False)
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

    parser = argparse.ArgumentParser('Combine the different widths results for experiments A and B')
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
    parser.add_argument('--table_format', choices=["wide", "long"], default="long")
    parser.add_argument("--yscale", choices=["log", "linear"], default="linear", help="the choice of the scale for y")
    parser.add_argument('--xlim', nargs='*', type=int, help='the bounds of the width')
    parser.add_argument('--vlim',  type=int, help='the number of variations to take')
    parser.set_defaults(cpu=False)
    parser.add_argument('dirs', nargs='*', help='the directory to process')
    parser.add_argument('--file', help='the csv data file')



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
        found = re.find(width_regexp, fname)
        return found[1]


    if args.file is not None and args.file.endswith(".csv"):
        process_csv(args.file, args)
        exit(0)


    Idx = pd.IndexSlice
    common_dir = os.path.commonpath(args.dirs)
    path_merge = os.path.join(common_dir, 'merge')
    fnames = []
    for dname in args.dirs:
        fnames.extend(glob.glob(os.path.join(dname, "**", "quant.csv"), recursive=True))

    unique_ids = set(list(map(get_parent, fnames)))
    for uid in unique_ids:  # for every f2 etc experiments
        path_output = os.path.join(path_merge, uid, "max")
        df_merge = pd.DataFrame(index=pd.Index([], name="var"))
        df_ref_merge = pd.DataFrame(index=pd.Index([], name="var"))
        os.makedirs(path_output, exist_ok=True)

        for directory in args.dirs:
            width = int(''.join([c for c in os.path.basename(directory.rstrip('/')) if c.isdigit()]))
            id_fnames = glob.glob(os.path.join(directory, "**", uid, "quant.csv"), recursive=True)


            for fn in id_fnames: # for all variations
                rid = int(''.join([c for c in os.path.basename(os.path.dirname(os.path.dirname(fn))) if c.isdigit()]))

                quant = read_csv(fn).sort_index(axis=1)

                if quant.columns.names != ["stat", "set", "layer"]:
                    # the order is
                    # perform pivot
                    quant = pd.melt(quant.reset_index(), id_vars="var").pivot(index="var", columns=["stat", "set", "layer"], values="value")
                fn_ref = os.path.join(os.path.dirname(fn), "stats_ref.csv")
                levels = list([[width]] +quant.columns.levels)
                quant.columns = pd.MultiIndex.from_product(levels,
                                                        names= ['width'] + quant.columns.names,
                                                        )
                quant_ref = pd.DataFrame()
                if os.path.isfile(fn_ref):  # if a reference file exists (i.e. experiment A)
                    quant_ref = pd.read_csv(fn_ref, index_col=[0,1]).transpose().sort_index(axis=1)
                    levels_ref = list([[width]] +quant_ref.columns.levels)
                    quant_ref.columns = pd.MultiIndex.from_product(levels_ref,
                                                            names= ['width'] + quant_ref.columns.names,
                                                            )
                else:  # the reference is at the layer 0 of the quant, that should be removed
                    quant_ref = quant.loc[:, Idx[:, :, :, 0]].droplevel("layer", axis=1).dropna()  # for all the  widths and all the vars
                    quant = quant.loc[:, Idx[:,  :, :, 1:]]

                quant_ref.index = [rid]

                col_order = quant.columns.names
                # quant_min=quant.min(axis=0).to_frame(name=rid).transpose()
                # quant_min.index.name = "var"
                quant_max=quant.max(axis=1, level=[0,1,2])  # take the max along all the levels except layer
                quant_min = quant_max.min(axis=0).to_frame(name=rid).transpose()

                if not rid in df_merge.index:  # id already in the index
                    df_merge = pd.concat([df_merge, quant_min], ignore_index=False, axis=0)
                    df_ref_merge = pd.concat([df_ref_merge, quant_ref], ignore_index=False, axis=0)
                elif width not in df_merge.columns.get_level_values("width"):
                    df_merge = pd.concat([df_merge, quant_min], ignore_index=False, axis=1)  # concatenate along the columns
                    df_ref_merge = pd.concat([df_ref_merge, quant_ref], ignore_index=False, axis=1)
                else:
                    df_merge.update(quant_min)  # update the df
                    df_ref_merge.update(quant_ref)  # update the df




                # df_merge = pd.concat([df_merge, quant_min], ignore_index=False, axis=0)
        df_merge.sort_index(axis=1, inplace=True)
        df_ref_merge.sort_index(axis=1, inplace=True)
        df_merge.to_csv(os.path.join(path_output, 'max.csv'))
        df_ref_merge.to_csv(os.path.join(path_output, 'ref.csv'))
        if args.vlim is not None:
            df_merge = df_merge.loc[1:args.vlim, :]
            df_ref_merge = df_ref_merge.loc[1:args.vlim, :]
                #log_mult = args_copy.log_mult#int(match.groups()[0])
                #columns=pd.MultiIndex.from_product([[log_mult], layers, stats], names=names)
                #epoch = checkpoint['epochs']
#            nlevel = len(quant.columns.levels)

                #level_width = C*[width]

        #df, epochs, args_entry = process_subdir(d, device)
        process_df(df_merge, path_output,  args, save=False, quant_ref=df_ref_merge)
        #process_epochs(epochs, dirname=d)


