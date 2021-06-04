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
        'lines.linewidth': 2
    }
)
# sns.set_theme()
# sns.set_style('whitegrid')
import glob



import torch.optim
import torch
import argparse
import utils



#from torchvision import models, datasets, transforms

try:
    from tqdm import tqdm
except:
    def tqdm(x): return x

def process_df(quant, dirname, quant_ref=None, args=None, args_model=None, save=True, split=False):

    global table_format

    col_names = ["experiment", "stat", "set", "width"]

    if quant.columns.names != col_names:
        # the order is
        # perform pivot
        quant = pd.melt(quant.reset_index(), id_vars="var").pivot(index="var", columns=col_names, values="value")
    quant =  utils.assert_col_order(quant, col_names, id_vars="var")
    keys = list(quant.columns.levels[0].sort_values())
    N_K = len(keys)  # the number of curves

    idx = pd.IndexSlice
    cols_error = idx[:, 'error', :, :]
    N_W = len(quant.columns.unique(level="width"))  # number of hidden layers
    # errors = quant["error"]
    # losses = quant["loss"]
    quant.drop("val", axis=1,level="set", inplace=True, errors='ignore')
    quant.drop(("test", "loss"), axis=1, inplace=True, errors='ignore')

    has_ref = quant_ref is not None
    WLIMS = {'MNIST': [0, 300], 'CIFAR-10': [0, 600]}


    output_root = dirname
    quant.sort_index(axis=1, inplace=True)
    quant.loc[:, cols_error] *= 100  # in %
    quant.groupby(level=["experiment", "stat", "set"], axis=1, group_keys=False).describe().to_csv(os.path.join(output_root, 'describe.csv'))

    if has_ref:
        quant_ref.sort_index(axis=1, inplace=True)
    #csvlosses.to_csv(os.path.join(output_root, 'losses.csv'))

    #errors.to_csv(os.path.join(output_root, 'errors.csv'))

    #quant_describe = pd.DataFrame(group.describe().rename(columns={'value': name}).squeeze()
    #                              for (name, group) in quant.groupby(level=["stat", "set"], axis=1))
    #quant_describe.to_csv(os.path.join(output_root, 'describe.csv'))


    df_reset = quant.reset_index()
    df_plot = pd.melt(df_reset, id_vars="var")#.query("layer>0")
    # df_plot_no_0 = df_plot.query('layer>0')
    # df_plot_0 = df_plot.query('layer==0')
    is_vgg = 'vgg' in dirname
    dataset = 'CIFAR-10' if 'cifar' in dirname else 'MNIST'
    # if args_model is not None:

    # if is_vgg:
        # xlabels=["0", "conv1", "conv2", "conv3", "conv4", "conv5", "conv6", "conv7", "conv8", "fc1", "fc2"]
    # else:
    # xlabels=[str(i) for i in range(N_L)]
    # xlabels =

    logstr = "_log" if args.yscale == "log" else ""
    zoomstr = "_zoom" if args.zoom else ""


    if N_K >= 3:
        pt = sns.color_palette(n_colors=N_K+1)  # remove green
        palette = pt[0:2] + pt[3:]
        palette_ref = pt[2:3]
    else:
        palette=sns.color_palette(n_colors=N_K)  # the N_K keys
        palette_ref = sns.color_palette(n_colors=3)[2:3]
    # if len(keys) <= 2:
        # palette=sns.color_palette(n_colors=2)[2-len(keys):] # the two experiments
    # else:
    if "dropout" in keys:
        keys = keys[::-1]

    palette=sns.color_palette(n_colors=len(keys))

    if not split:
        fig, axes = plt.subplots(4, 1, figsize=(4, 16), sharex=False)
    # sns.set(font_scale=1,rc={"lines.linewidth":3})

    # fig.suptitle("{} {}".format('VGG' if is_vgg else 'FCN', dataset.upper()))
    k = 0

    Ts = { 0: 0, 1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 9: 2.262}
    quant_log = np.log10(quant)
    df_ci = quant.describe()

    df_ci.loc["ymax", :] =  [mean + Ts[int(n-1)] / np.sqrt(n) * std for (mean, std, n) in zip(df_ci.loc["mean",  :], df_ci.loc["std", :], df_ci.loc["count", :])]
    df_ci.loc["ymin", :] = [mean - Ts[int(n-1)] / np.sqrt(n) * std for (mean, std, n) in zip(df_ci.loc["mean",  :], df_ci.loc["std", :], df_ci.loc["count", :])]

    # quant_log = quant.log()
    # df_ci_log = quant_log.describe()
    #rp.set_axis_labels("layer", "Loss", labelpad=10)
    #quant.loc[1, Idx["loss", :, 0]].lineplot(x="layer_ids", y="value", hue="")

    df_ci = quant.describe()
    df_ci_log = quant_log.describe()

    df_ci.loc["ymax", :] =  [mean + Ts[int(n-1)] / np.sqrt(n) * std for (mean, std, n) in zip(df_ci.loc["mean",  :], df_ci.loc["std", :], df_ci.loc["count", :])]
    df_ci.loc["ymin", :] = [mean - Ts[int(n-1)] / np.sqrt(n) * std for (mean, std, n) in zip(df_ci.loc["mean",  :], df_ci.loc["std", :], df_ci.loc["count", :])]
    #rp.set_axis_labels("layer", "Loss", labelpad=10)
    df_ci_log.loc["ymax", :] =  [mean + Ts[int(n-1)] / np.sqrt(n) * std for (mean, std, n) in zip(df_ci_log.loc["mean",  :], df_ci_log.loc["std", :], df_ci_log.loc["count", :])]
    df_ci_log.loc["ymin", :] = [mean - Ts[int(n-1)] / np.sqrt(n) * std for (mean, std, n) in zip(df_ci_log.loc["mean",  :], df_ci_log.loc["std", :], df_ci_log.loc["count", :])]
    # lines = []
    for (stat, setn) in [('loss', 'train'), ('error', 'test')]:
        for exp in keys:
        # for i, stat in enumerate(["loss","error" ]):
            # for j, setn in enumerate(["train","test"]):

            # if stat == "loss" and setn=="test":
                # continue
            # if stat == "error" and setn=="train":
                # continue
            # axes[k] = rp.axes[j,i]
            if split:
                fig, ax = plt.subplots(1, 1, figsize=(4, 4), sharex=False)
            else:
                ax = axes.flatten()[k]

            log_plot = args.yscale == "log" and setn == "train"

            if log_plot:  # take the log values to plot
                df_plot = quant_log.loc[:, Idx[exp, stat, setn, :]]
                df_ci_plot = df_ci_log
            else:
                df_plot = quant.loc[:, Idx[exp, stat, setn, :]]#.min(axis=0).to_frame(name="value")
                df_ci_plot = df_ci


            if exp == 'A':
                wlim = WLIMS[dataset]
            else:
                wlim = None
            if wlim is not None:
                df_plot = df_plot.loc[:, Idx[:, :, :, wlim[0]:wlim[1]]]#.min(axis=0).to_frame(name="value")
            # else:
                # df_plot = df_plot.loc[:, Idx[, :]]#.min(axis=0).to_frame(name="value")


            df_plot = pd.melt(df_plot.reset_index(), id_vars="var")
            xs = df_plot["width"].unique().astype(np.int32)
            df_ci_plot = df_ci_plot.loc[:, Idx[exp, stat, setn, xs]]
            lp = sns.lineplot(
                #data=rel_losses.min(axis=0).to_frame(name="loss"),
                # data=df_plot_rel if not is_vgg else df_plot_rel.pivot(index="steps", columns=col_order).min(axis=0).to_frame(name="value"),
                data=df_plot,
                #hue="width",
                hue="experiment",
                hue_order=keys,
                x="width",
                y="value",
                legend=None,
                # style='set',
                ci=None,
                palette=palette,
                #style='layer',
                markers=False,
                ax=ax,
                dashes=True,
                # linewidth=3.,
                #legend_out=True,
                #y="value",
            )

            # xs = df_plot["width"]
            ax.fill_between(xs, df_ci_plot.loc["ymax",:].values, df_ci_plot.loc["ymin", :].values, color=ax.lines[-1].get_color(), alpha=0.3)
            # lp.set(xticks=range(0, len(xlabels)))
            # rp.set_xticklabels(xlabels)
            # rp.axes[0,0].locator_params(axis='x', nbins=len(xlabels))
            # lp.set_xticklabels(xlabels)#, rotation=40*(is_vgg))
            # else:
                # lp.set_xticklabels(len(xlabels)*[None])
            if not split:
                ax.set_title(dataset)
            # ax.set_title("{} {}{}".format(setn.title()+(setn=="train")*"ing", stat.title(), " (%)" if stat=="error" else ''))
            # ylabel = stat if stat == "loss" else "error (%)"
            ax.set_xlabel("width")
            ax.set_ylabel(None)
            # ax.tick_params(labelbottom=True)
            # idx = 0
            # widths = quant.loc[:,
            # 'A'].columns.get_level_values("width").unique().
            # eps = 0.0005
            # while idx < len(widths) and quant.loc[:, Idx['A', stat, setn, widths[idx]]].min() > eps:
                # idx += 1
                # )

                # if has_ref:
                    # df_ref = quant_ref.loc[:, Idx[:, stat,setn]]

            if has_ref:
                if log_plot:
                    df_ref = np.log10(quant_ref.loc[:, Idx[df_plot['width'], stat, setn]])
                else:
                    df_ref = quant_ref.loc[:, Idx[df_plot['width'], stat, setn]]

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

                for l in ax.lines[-1:]:
                    l.set_linestyle(':')


        # if has_ref:
            # # data_ref  = quant_ref[stat, setn].reset_index()
            if log_plot:
                # # ax.set_yticks([10**(p) for p in ax.get_yticks()])
                # # ax.set_ylabel([10^
                # ax.get_yaxis().set_major_locator(matplotlib.ticker.LogLocator())
                # # ax.yaxis.get_major_locator().tick_values(vals[0], vals[-1])

                ax.get_yaxis().get_major_formatter().set_useMathText(True)

                ax.get_yaxis().set_major_formatter(lambda x, pos:  "$10^{" + f"{int(x)}" + "}$")

            # ax.axline((0,quant_ref[stat, setn][0]), (1, quant_ref[stat, setn][0]),  ls=":", zorder=2, c='g')  # for the mean
            # y1 = quant_ref.loc['mean', (stat, setn)] + quant_ref.loc['ci95', (stat, setn)]
            # y2 = quant_ref.loc['mean', (stat, setn)] - quant_ref.loc['ci95', (stat, setn)]
            # ax.axhspan(y1, y2, facecolor='g', alpha=0.5)
            # data_ref.index = pd.Index(range(len(data_ref)))
                # ax=ax,
        # lines.extend(ax.lines)
            # if log_plot:
                # df_plot = quant_log.loc[:, Idx[:, stat, setn, :]]
                # df_ci_plot = df_ci_log
            # else:
                # df_plot = quant.loc[:, Idx[:, stat, setn, :]]#.min(axis=0).to_frame(name="value")
                # df_ci_plot = df_ci

            if split:
                # ax.set_title(dataset)
                # if k == 1:
                labels=[exp] + ["ref."]
                if setn == "test":  # reset the name (not log)
                    logstr = ""
                loc = "upper right"
                bbox = (0.9,0.7) if (exp == 'B' and dataset == "CIFAR-10" and log_plot) else (0.9, 0.9)
                # if exp == 'B':
                fig.legend(handles=ax.lines, labels=labels,
                            # title="Exp.",
                            loc=loc, borderaxespad=0, bbox_to_anchor=bbox)#, bbox_transform=fig.transFigure)
                # lines = []

                # fig.tight_layout()
                plt.margins()

                plt.savefig(fname=os.path.join(output_root, f"{exp}_{setn}_{stat}{logstr}{zoomstr}.pdf"), bbox_inches='tight')

            k += 1

    # fig.subplots_adjust(top=0.85)
    # if is_vgg:
    if not split:
        labels=keys + ["ref."]
        fig.legend(handles=ax.lines, labels=labels,
                  # title="Exp.",
                   loc="upper right", borderaxespad=0, bbox_to_anchor=(0.9,0.9))#, bbox_transform=fig.transFigure)
        fig.tight_layout()
        # plt.margins()
        fig.savefig(fname=os.path.join(output_root, f"train_loss_test_error{logstr}{zoomstr}.pdf"), bbox_inches='tight')
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

            # df_plot = quant.loc[:, Idx[:, stat, setn, :]].min(axis=0).to_frame(name="value")
            df_plot = quant.loc[:, Idx[:, stat, setn, :]]#.min(axis=0).to_frame(name="value")
            df_plot = pd.melt(df_plot.reset_index(), id_vars="var")
            lp = sns.lineplot(
                #data=rel_losses.min(axis=0).to_frame(name="loss"),
                # data=df_plot_rel if not is_vgg else df_plot_rel.pivot(index="steps", columns=col_order).min(axis=0).to_frame(name="value"),
                data=df_plot,
                #hue="width",
                hue="experiment",
                hue_order=keys,
                x="width",
                y="value",
                legend=None,
                estimator=None,
                # style='set',
                ci=None,
                palette=palette,
                #style='layer',
                markers=False,
                ax=ax,
                dashes=True,
                #legend_out=True,
                #y="value",
            )

            # Ts = { 0: 0, 2: 4.303, 3: 3.182, 4: 2.776, 9: 2.262}
            # df_ci = quant.loc[:, Idx[:, stat, setn, :]].describe()

            # df_ci.loc["ymax", :] =  [mean + Ts[n-1] / np.sqrt(n) * std for (mean, std, n) in zip(df_ci.loc["mean",  :], df_ci.loc["std", :], df_ci.loc["count", :])]
            # df_ci.loc["ymin", :] = [mean - Ts[n-1] / np.sqrt(n) * std for (mean, std, n) in zip(df_ci.loc["mean",  :], df_ci.loc["std", :], df_ci.loc["count", :])]

            # ax.fill_between(df_plot["layers"], df_ci["ymax"], df_ci["ymin"])
            # lp.set(xticks=range(0, len(xlabels)))
            # rp.set_xticklabels(xlabels)
            # rp.axes[0,0].locator_params(axis='x', nbins=len(xlabels))
            # rp.axes[0,1].locator_params(axis='x', nbins=len(xlabels))

            # lp.set_xticklabels(xlabels)#, rotation=40*(is_vgg))
            if not split:
                ax.set_title(dataset)
            # ax.set_title("{} {}{}".format(setn.title()+(setn=="train")*'ing', stat.title(), " (%)" if stat=="error" else ''))
            # ylabel = stat if stat == "loss" else "error (%)"
            ax.set_xlabel("width")
            ax.set_ylabel(None)

            if has_ref:
                # data_ref  = quant_ref[stat, setn].reset_index()

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

                for l in ax.lines[-1:]:
                    l.set_linestyle(':')
                # ax.axline((0,quant_ref[stat, setn][0]), (1,quant_ref[stat, setn][0]), ls=":", zorder=2, c='g')
                # data_ref.index = pd.Index(range(len(data_ref)))
                # sns.lineplot(
                    # data=data_ref,  # repeat the datasaet N_L times
                    # ax=ax,
            if setn == "train":
                ax.set_yscale(args.yscale)
                    # # x=range(len(data_ref)),
                    # # y="value",
                    # # xc np.tile(np.linspace(1, N_L, num=N_L), 2),
                    # # x='',
                    # # hue='r',
                    # # color='red',
                    # palette=['red'],
                    # # style='set',
                    # # x='index',
                    # # dashes=True,
                    # legend=False,
                    # # y="value"
                # )

                # for ax in ax.lines[-1:]:  # the last two
                    # ax.set_linestyle('--')
            k += 1


    # fig.subplots_adjust(top=0.85)
    # if is_vgg:
    labels=keys + ["ref."]
    fig.legend(handles=ax.lines, labels=keys,
               #title="Exp.",
               loc="upper right", bbox_to_anchor=(0.9,0.9),borderaxespad=0)#, bbox_transform=fig.transFigure)
    plt.margins()
    plt.savefig(fname=os.path.join(output_root, f"error_train{logstr}.pdf"), bbox_inches='tight')

    if "B" in keys:
        df_B = quant["B"]
    elif "B2" in keys:
        df_B = quant["B2"]
    else:
        return
    n_draws = len(df_B.index)
    # vary_draw=copy.deepcopy(df_B)
    df_B_plot = pd.melt(df_B.reset_index(), id_vars="var")
    cp = sns.FacetGrid(
        data=df_B_plot,
        # hue="experiment",
        # hue_order=["A", "B"],
        col="stat",
        col_order=["loss", "error"],
        row="set",
        row_order=["train", "test"],
        # x="layer",
        # y="value",
        # kind='line',
        # legend="full",
        # style='set',
        # ci='sd',
        # palette=palette,
        #style='layer',
        # markers=False,
        # dashes=True,
        #legend_out=True,
        # facet_kws={
        sharey= False,
        sharex= True,
        #y="value",
    )
    styles=['dotted', 'dashed', 'dashdot',  'solid']
    # for i_k, k in enumerate([10, 50, 100, 200]):
    draws = len(df_B.index)
    df_bound = pd.DataFrame(columns=df_B.columns)
    # df_bound.columns = df_B.columns
    # for k in range(1, draws+1):
        # # df_cut = pd.melt(df_B[:k].reset_index(), id_vars="draw")
        # df_bound.loc[k, :] = df_B[:k].min(axis=0)
        # # idx_min = df_cut.query('stat=="loss"idxmin")
    # fig, axes= plt.subplots(2,2,figsize=(12,12), sharex=True)
    # for i, stat in enumerate(["loss", "error"]):
        # for j, setn in enumerate(["train", "test"]):
            # df_bound_plot = df_bound[stat,setn].max(axis=1)
            # ax=axes[i,j]
            # ax.set_title("{} {}".format(setn.title(), stat.title()))
            # sns.lineplot(
                # data=df_bound_plot,
                # ax=ax,
            # )
                # # cp.axes[j,i].set_title("{} {}".format(setn.title(), stat.title()))
    # plt.savefig(fname=os.path.join(output_root, "comp_draws.pdf"), bbox_inches='tight')

    plt.close('all')
                # ylabel = stat if stat == "loss" else "error (%)"
                # cp.axes[j,i].set_ylabel(ylabel)
                # cp.axes[j,i].set_xlabel("layer index l")

                # df_cut_plot = pd.melt(df_cut_min.query(f'stat=="{stat}" & set=="{setn}"'))
                # if has_ref:
                    # data_ref  = quant_ref[stat, setn].reset_index()

                    # data_ref.index = pd.Index(range(len(data_ref)))
                # sns.lineplot(
                    # data=df_cut_plot,  repeat the datasaet N_L times
                    # ax=cp.axes[j,i],
                    # x=range(len(data_ref)),
                    # y="value",
                    # xc np.tile(np.linspace(1, N_L, num=N_L), 2),
                    # x='layer',
                    # hue='r',
                    # color='red',
                    # palette=['red'],
                    # style='set',
                    # x='index',
                    # dashes=True,
                    # legend=False,
                    # y="value"
                # )

                # for ax in cp.axes[j,i].lines[-1:]:  the last two
                    # ax.set_linestyle(styles[i_k])




def process_csv(file_csv):
    '''Read and process a previously computed result stored inside a checkpoint'''

    idx = pd.IndexSlice
    quant = pd.read_csv(file_csv, header=[0,1], index_col=0)
    process_df(quant, os.path.dirname(file_csv))
    return

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

if __name__ == '__main__':

    torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser('Evaluating a copy of a classifier with removed units')
    parser_device = parser.add_mutually_exclusive_group()
    parser_device.add_argument('--cpu', action='store_true', dest='cpu', help='force the cpu model')
    parser_device.add_argument('--cuda', action='store_false', dest='cpu')
    #parser.add_argument('--end_layer', type=int, help='if set the maximum layer for which to compute the separation (forward indexing)')
    parser.add_argument('--table_format', choices=["wide", "long"], default="long")
    parser.add_argument('--experiments', nargs='*', default=['A', 'B'], help='whitelist for the experiments to cat')
    parser.add_argument('--yscale', choices=["linear", "log"], default='linear', help='the scale for the y axis')
    parser.add_argument('--vlim',  default=3, type=int, help='the number of variations to take')
    parser.add_argument('--zoom',  action='store_true',  help='if we zoom on the graph for A')
    parser.add_argument('dirs', nargs='*', help='the directories to process')
    parser.add_argument('--split', action='store_true', default=True, help='split the err/loss figures in two')
    parser.set_defaults(cpu=False)



    args = parser.parse_args()
    table_format = args.table_format

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    #device = torch.device('cpu')


    dtype = torch.float

    def get_parent(path):
        return os.path.basename(os.path.dirname(path))

    def get_grand_parent(path):
        return os.path.dirname(os.path.dirname(path.rstrip(os.sep)))

    for directory in args.dirs:

        lst_file = glob.glob(os.path.join(directory, "**", "max.csv"), recursive=True)  # all the saved results
        roots = set(map(get_grand_parent, lst_file))

        for root in roots:
            id_lst_file = glob.glob(os.path.join(root, "**", "max.csv"), recursive=True)
            df_bundle = pd.DataFrame()
            quant_ref = None
            args_model = None
            for f in id_lst_file:
                experiment = get_parent(f)
                if not experiment in args.experiments:
                    continue
                # try:
                    # f_model = torch.load(os.path.join(os.path.dirname(f), "checkpoint.pth"), map_location=device)
                # except IOError as e:
                    # print(f"error {e} for {f}")
                    # f_model = None
                # if f_model:
                    # chkpt_model =torch.load(f_model, map_location=device)
                    # # args_chkpt  = chkpt['args']
                    # args_model = chkpt_model['args']

                Idx = pd.IndexSlice

                # quant = chkpt['quant'].sort_index(axis=1)
                quant = pd.read_csv(f, header=[0,1,2], index_col=0)

# dirname = os.path.dirname(file_csv)
                            # quant.index.rename("var", inplace=True)
                df_lst = [quant]
                fn_ref = os.path.join(os.path.dirname(f), "ref.csv")
                if os.path.isfile(fn_ref):
                    new_ref = pd.read_csv(fn_ref, header=[0,1,2], index_col=0)
                    # if quant_ref is None:
                        # quant_ref = pd.read_csv(fn_ref, header=[0,1,2], index_col=0)
                    # else:
                        # new_ref = pd.read_csv(fn_ref, headr=[0,1,2], index_col=0)
                        # quant_ref.update(pd.read_csv(fn_ref, header=[0,1,2], index_col=0))

                if new_ref is not None:
                    df_lst += [new_ref]


                check_columns(*df_lst)

                if new_ref is not None:
                    if quant_ref is None or len(new_ref.columns) > len(quant_ref.columns):
                        quant_ref = new_ref
                # if 'B' in experiment or 'dropout' in experiment:
                    # quant_ref = quant.loc[:, Idx[:, :, 0]]  #layer = 0 is the reference
                    # quant = quant.loc[:, Idx[:, :, 1:]]  # only keep layers > 0

                ncol = len(quant.columns)
                levels = list([ncol*[experiment]] +[quant.columns.get_level_values(i) for i in range(quant.columns.nlevels)])
                # quant.columns = pd.MultiIndex.from_product(levels,
                                                        # names= ['experiment'] + quant.columns.names,
                                                        # )
                quant.columns = pd.MultiIndex.from_arrays(levels,
                                        names= ['experiment'] + quant.columns.names,
                                          )
                                                        # )

                df_bundle = pd.concat([df_bundle, quant], ignore_index=False, axis=1)
                #df_bundle.loc[:, (log_mult, layers, 'loss')] = quant.xs('loss', level=1, axis=1)
                #df_bundle.loc[Idx[:, (log_mult, layers, 'error')]] = quant.xs('error', level=1, axis=1)
                #epochs[idx_entry] = epoch

            output_root = os.path.join(root, f"widths_" + "_".join(args.experiments))
            os.makedirs(output_root, exist_ok=True)


            df_bundle.sort_index(axis=1, inplace=True)
            df_bundle.to_csv(os.path.join(output_root, 'widths.csv'))
            if quant_ref is not None:
                quant_ref.to_csv(os.path.join(output_root, 'quant_ref.csv'))
            if not df_bundle.empty:
                if args.vlim is not None:
                    df_bundle = df_bundle.loc[1:args.vlim, :]
                    quant_ref = quant_ref.loc[1:args.vlim, :]
                process_df(df_bundle, output_root, quant_ref, args=args, args_model=args_model, split=args.split)

    sys.exit(0)





