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
sns.set(
    font_scale=1.5,
    style="whitegrid",
    rc={
    'text.usetex' : False,
        'lines.linewidth': 3
    }
)
import glob


import torch.optim
import torch
import argparse
import utils

def process_df(quant, dirname, stats_ref=None, args=None, args_model=None, save=True):

    global table_format

    col_names = ["experiment", "stat", "set", "layer"]
    quant =  utils.assert_col_order(quant, col_names, id_vars="draw")
    keys = list(quant.columns.levels[0].sort_values())

    output_root = os.path.join(dirname, f"meta_" + "_".join(keys))
    os.makedirs(output_root, exist_ok=True)
    idx = pd.IndexSlice
    cols_error = idx[:, 'error', :, :]
    N_L = len(quant.columns.unique(level="layer"))  # number of hidden layers
    # errors = quant["error"]
    # losses = quant["loss"]
    quant.drop("val", axis=1,level="set", inplace=True, errors='ignore')
    quant.drop(("test", "loss"), axis=1, inplace=True, errors='ignore')

    if save:
        quant.to_csv(os.path.join(output_root, 'merge.csv'))
        if stats_ref is not None:
            stats_ref.to_csv(os.path.join(output_root, 'stats_ref.csv'))


    quant.sort_index(axis=1, inplace=True)
    quant.loc[:, cols_error] *= 100  # in %
    quant.groupby(level=["experiment", "stat", "set"], axis=1, group_keys=False).describe().to_csv(os.path.join(output_root, 'describe.csv'))

    df_reset = quant.reset_index()
    df_plot = pd.melt(df_reset, id_vars='draw')#.query("layer>0")
    # df_plot_no_0 = df_plot.query('layer>0')
    quant_ref = None
    if stats_ref is not None:
        N_S = len(stats_ref)
        quant_ref_merge = pd.DataFrame()
        stats_ref.loc["error"] = stats_ref["error"].values * 100
        for key in keys:
            quant_ref_merge = pd.concat([quant_ref_merge, quant_ref])
            # N_S_key = len(quant[key].columns.get_level_values("stat").unique())
            N_L_key = len(quant[key].columns.get_level_values("layer").unique())
            quant_ref_key = stats_ref.iloc[np.tile(np.arange(N_S).reshape(N_S, 1), (N_L_key)).ravel()].to_frame(name="value").droplevel("layer")
            quant_ref_merge = pd.concat([quant_ref_merge, quant_ref_key])

        quant_ref = stats_ref.iloc[np.repeat(np.arange(N_S), (N_L))].to_frame(name="value").droplevel("layer").value
        try:
            utils.to_latex(output_root, (quant-quant_ref_merge.value.values).abs(), table_format, key_err="error")
        except:
            pass

    is_vgg = 'vgg' in dirname
    dataset = 'CIFAR10' if 'cifar' in dirname else 'MNIST'
    # if args_model is not None:
    xlabels=[str(i) for i in range(N_L)]

    palette=sns.color_palette(n_colors=len(keys))  # the two experiments
    fig, axes = plt.subplots(2, 1, figsize=(4, 8), sharex=False)
    # sns.set(font_scale=1,rc={"lines.linewidth":3})

    # fig.suptitle("{} {}".format('VGG' if is_vgg else 'FCN', dataset.upper()))
    k = 0

    #rp.set_axis_labels("layer", "Loss", labelpad=10)
    #quant.loc[1, Idx["loss", :, 0]].lineplot(x="layer_ids", y="value", hue="")
    for i, stat in enumerate(["loss","error" ]):
        for j, setn in enumerate(["train","test"]):
            if stat == "loss" and setn=="test":
                continue
            if stat == "error" and setn=="train":
                continue
            # axes[k] = rp.axes[j,i]
            ax = axes.flatten()[k]

            df_plot = quant.loc[:, Idx[:, stat, setn, :]].min(axis=0).to_frame(name="value")
            lp = sns.lineplot(
                #data=rel_losses.min(axis=0).to_frame(name="loss"),
                # data=df_plot_rel if not is_vgg else df_plot_rel.pivot(index="draw", columns=col_order).min(axis=0).to_frame(name="value"),
                data=df_plot,
                #hue="width",
                hue="experiment",
                hue_order=keys,
                x="layer",
                y="value",
                legend=None,
                # style='set',
                ci='sd',
                palette=palette,
                #style='layer',
                markers=False,
                ax=ax,
                dashes=True,
                # linewidth=3.,
                #legend_out=True,
                #y="value",
            )
            lp.set(xticks=range(0, len(xlabels)))
            # rp.set_xticklabels(xlabels)
            # rp.axes[0,0].locator_params(axis='x', nbins=len(xlabels))
            # rp.axes[0,1].locator_params(axis='x', nbins=len(xlabels))

            # if k == 1:
            # if k==1:
            lp.set_xticklabels(xlabels)#, rotation=40*(is_vgg))
            # else:
                # lp.set_xticklabels(len(xlabels)*[None])
            ax.set_title("{} {}{}".format(setn.title()+(setn=="train")*"ing", stat.title(), " (%)" if stat=="error" else ''))
            # ylabel = stat if stat == "loss" else "error (%)"
            ax.set_xlabel("layer index l")
            ax.set_ylabel(None)
            # ax.tick_params(labelbottom=True)


            if quant_ref is not None:
                # data_ref  = quant_ref[stat, setn].reset_index()

                ax.axline((0,quant_ref[stat, setn][0]), slope=0, ls=":", zorder=2, c='g')
                # for ax in ax.lines[-1:]:  # the last two
                    # ax.set_linestyle('--')
            k += 1


    # fig.subplots_adjust(top=0.85)
    # if is_vgg:
    labels=keys + ["Ref."]
    fig.legend(handles=ax.lines, labels=labels, title="Exp.",  loc="upper right", borderaxespad=0, bbox_to_anchor=(0.9,0.9))#, bbox_transform=fig.transFigure)
    fig.tight_layout()
    plt.margins()
    fig.savefig(fname=os.path.join(output_root, "train_loss_test_error.pdf"), bbox_inches='tight')
    k=0
    # sns.set(font_scale=1,rc={"lines.linewidth":3})

    fig, axes = plt.subplots(1, 1, figsize=(4, 4), sharex=False)
    # fig.suptitle("{} {}".format('VGG' if is_vgg else 'FCN', dataset.upper()))

    for i, stat in enumerate(["error"]):
        for j, setn in enumerate(["train"]):
            if stat == "loss" and setn=="test":
                continue
            if stat=="error" and setn=="test":
                continue
            # axes[k] = rp.axes[j,i]
            ax = axes

            df_plot = quant.loc[:, Idx[:, stat, setn, :]].min(axis=0).to_frame(name="value")
            lp = sns.lineplot(
                #data=rel_losses.min(axis=0).to_frame(name="loss"),
                # data=df_plot_rel if not is_vgg else df_plot_rel.pivot(index="draw", columns=col_order).min(axis=0).to_frame(name="value"),
                data=df_plot,
                #hue="width",
                hue="experiment",
                hue_order=keys,
                x="layer",
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
            lp.set(xticks=range(0, len(xlabels)))
            # rp.set_xticklabels(xlabels)
            # rp.axes[0,0].locator_params(axis='x', nbins=len(xlabels))
            # rp.axes[0,1].locator_params(axis='x', nbins=len(xlabels))

            lp.set_xticklabels(xlabels)#, rotation=40*(is_vgg))
            ax.set_title("{} {}{}".format(setn.title()+(setn=="train")*'ing', stat.title(), " (%)" if stat=="error" else ''))
            # ylabel = stat if stat == "loss" else "error (%)"
            ax.set_xlabel("layer index l")
            ax.set_ylabel(None)

            if quant_ref is not None:
                # data_ref  = quant_ref[stat, setn].reset_index()

                ax.axline((0,quant_ref[stat, setn][0]), slope=0, ls=":", zorder=2, c='g')

            k += 1


    labels=keys + ["Ref."]
    fig.legend(handles=ax.lines, labels=keys, title="Exp.", loc="upper right", bbox_to_anchor=(0.9,0.9),borderaxespad=0)#, bbox_transform=fig.transFigure)
    plt.margins()
    plt.savefig(fname=os.path.join(output_root, "error_train.pdf"), bbox_inches='tight')

    if "B" in keys:
        df_B = quant["B"]
    elif "B2" in keys:
        df_B = quant["B2"]
    else:
        return
    n_draws = len(df_B.index)
    # vary_draw=copy.deepcopy(df_B)
    df_B_plot = pd.melt(df_B.reset_index(), id_vars="draw")
    cp = sns.FacetGrid(
        data=df_B_plot,
        # hue="experiment",
        # hue_order=["A", "B"],
        col="stat",
        col_order=["loss", "error"],
        row="set",
        row_order=["train", "test"],
        sharey= False,
        sharex= True,
        #y="value",
    )
    styles=['dotted', 'dashed', 'dashdot',  'solid']
    # for i_k, k in enumerate([10, 50, 100, 200]):
    draws = len(df_B.index)
    df_bound = pd.DataFrame(columns=df_B.columns)
    # df_bound.columns = df_B.columns
    for k in range(1, draws+1):
        # df_cut = pd.melt(df_B[:k].reset_index(), id_vars="draw")
        df_bound.loc[k, :] = df_B[:k].min(axis=0)
        # idx_min = df_cut.query('stat=="loss"idxmin")
    fig, axes= plt.subplots(2,2,figsize=(12,12), sharex=True)
    for i, stat in enumerate(["loss", "error"]):
        for j, setn in enumerate(["train", "test"]):
            df_bound_plot = df_bound[stat,setn].max(axis=1)
            ax=axes[i,j]
            ax.set_title("{} {}".format(setn.title(), stat.title()))
            sns.lineplot(
                data=df_bound_plot,
                ax=ax,
            )
                # cp.axes[j,i].set_title("{} {}".format(setn.title(), stat.title()))
    plt.savefig(fname=os.path.join(output_root, "comp_draws.pdf"), bbox_inches='tight')

    plt.close('all')
                # ylabel = stat if stat == "loss" else "error (%)"



def process_csv(file_csv):
    '''Read and process a previously computed result stored inside a checkpoint'''

    idx = pd.IndexSlice
    quant = pd.read_csv(file_csv, header=[0,1], index_col=0)
    process_df(quant, os.path.dirname(file_csv))
    return

if __name__ == '__main__':

    torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser('Evaluating a copy of a classifier with removed units')
    parser_device = parser.add_mutually_exclusive_group()
    parser_device.add_argument('--cpu', action='store_true', dest='cpu', help='force the cpu model')
    parser_device.add_argument('--cuda', action='store_false', dest='cpu')
    #parser.add_argument('--end_layer', type=int, help='if set the maximum layer for which to compute the separation (forward indexing)')
    parser.add_argument('--table_format', choices=["wide", "long"], default="long")
    parser.add_argument('--experiments', nargs='*', default=['A', 'B'], help='whitelist for the experiments to cat')
    parser.add_argument('dirs', nargs='*', help='the directories to process')
    parser.set_defaults(cpu=False)



    args = parser.parse_args()
    table_format = args.table_format

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    #device = torch.device('cpu')


    dtype = torch.float
    num_gpus = torch.cuda.device_count()

    def get_parent(path):
        return os.path.basename(os.path.dirname(path))

    def get_grand_parent(path):
        return os.path.dirname(os.path.dirname(path.rstrip(os.sep)))

    for directory in args.dirs:

        lst_file = glob.glob(os.path.join(directory, "**", "quant.csv"), recursive=True)  # all the saved results
        roots = set(map(get_grand_parent, lst_file))

        for root in roots:
            id_lst_file = glob.glob(os.path.join(root, "**", "quant.csv"), recursive=True)
            df_bundle = pd.DataFrame()
            stats_ref = None
            args_model = None
            for f in id_lst_file:
                experiment = get_parent(f)
                if not experiment in args.experiments:
                    continue

                Idx = pd.IndexSlice

                # quant = chkpt['quant'].sort_index(axis=1)
                quant = pd.read_csv(f, header=[0,1,2], index_col=0)
                nlevels = quant.columns.nlevels
                layer_idx = quant.columns.names.index("layer")
                stat_idx = quant.columns.names.index("stat")
                if quant.columns.get_level_values(layer_idx).dtype != int:  # 0 are the layers
                    new_layer_lvl = list(map(int, quant.columns.get_level_values(layer_idx)))
                    levels = [quant.columns.get_level_values(i) if i != layer_idx else new_layer_lvl for i in range(nlevels)]
                    cols = pd.MultiIndex.from_arrays(levels, names=quant.columns.names)
                    quant.columns = cols

                quant.index.rename("draw", inplace=True)
                quant.sort_index(axis=1, inplace=True)

                if 'B' in experiment:
                    stats_ref = quant.loc[1, Idx[:, :, 0]]  #layer = 0 is the reference
                    quant = quant.loc[:, Idx[:, :, 1:]]  # only keep layers > 0

                levels = list([[experiment]] +[quant.columns.get_level_values(i).unique() for i in range(quant.columns.nlevels)])
                quant.columns = pd.MultiIndex.from_product(levels,
                                                        names= ['experiment'] + quant.columns.names,
                                                        )

                df_bundle = pd.concat([df_bundle, quant], ignore_index=False, axis=1)


            df_bundle.sort_index(axis=1, inplace=True)
            if not df_bundle.empty:
                process_df(df_bundle, root, stats_ref, args_model=args_model)

    sys.exit(0)





