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
sns.set(font_scale=1.5, style="whitegrid", rc={'text.usetex' : False, 'lines.linewidth': 3})
# sns.set_theme()
# sns.set_style('whitegrid')
import glob

import math


import torch.optim
import torch
import argparse
import utils



def process_checkpoint(checkpoint):
    '''Read and process a previously computed result stored inside a checkpoint'''

    quant = checkpoint['quant']
    args = checkpoint['args']
    process_df(quant, args.path_output)
    return

def process_df(quant, dirname, uid, args=None, args_model=None, save=True):

    global table_format

    if quant.columns.names != ["stat", "set", "layer", "width"]:
        # the order is
        # perform pivot
        quant = pd.melt(quant.reset_index(), id_vars="draw").pivot(index="draw", columns=["stat", "set", "layer", "width"], values="value")

    output_root = os.path.join(dirname, f"meta_{uid}")
    os.makedirs(output_root, exist_ok=True)
    idx = pd.IndexSlice
    cols_error = idx['error', :, :, :]
    N_L = len(quant.columns.unique(level="layer")) -1 # number of hidden layers
    errors = quant["error"]
    losses = quant["loss"]

    if save:
        quant.to_csv(os.path.join(output_root, 'quant.csv'))


    quant.sort_index(axis=1, inplace=True)
    quant.loc[:, cols_error] *= 100  # in %
    quant.groupby(level=["stat", "set"], axis=1, group_keys=False).describe().to_csv(os.path.join(output_root, 'describe.csv'))
    #csvlosses.to_csv(os.path.join(output_root, 'losses.csv'))
    #errors.to_csv(os.path.join(output_root, 'errors.csv'))

    #quant_describe = pd.DataFrame(group.describe().rename(columns={'value': name}).squeeze()
    #                              for (name, group) in quant.groupby(level=["stat", "set"], axis=1))
    #quant_describe.to_csv(os.path.join(output_root, 'describe.csv'))

    #fig=plt.figure()
    #f, axes = plt.subplots(1, 2, figsize=[10., 5.])

    df_reset = quant.reset_index()
    df_plot = pd.melt(df_reset, id_vars='draw')#.query("layer>0")
    df_plot_no_0 = df_plot.query('layer>0')
    df_plot_0 = df_plot.query('layer==0')
    #relative quantities
    quant_ref = quant.loc[1, Idx[:, :, 0, :]].droplevel("layer")  # for all the  widths
    N_S = len(quant_ref)
    quant_ref_val = quant_ref.iloc[np.repeat(np.arange(N_S), N_L)].values
    quant_rel = (quant.loc[:, Idx[:, :, 1:]] - quant_ref_val).abs()
    #quant_plus = quant.loc[:, Idx[:, :, 1:]] + quant_ref + 1e-10
    #quant_rel /= quant_plus
    #quant_rel *= 2

    # utils.to_latex(output_root, quant.loc[:, Idx[:, :, 1:]], table_format, key_err="error")
    utils.to_latex(output_root, quant_rel, table_format, key_err="error")

    df_reset_rel = quant_rel.reset_index()
    df_plot_rel = pd.melt(df_reset_rel, id_vars="draw")

    # palette=sns.color_palette(n_colors=2)  # the two datasets
    palette=sns.color_palette(n_colors=N_L)  # the N_L layers
    # bp = sns.catplot(
        # data = df_plot.query('layer > 0'),
        # #col='log_mult',
        # hue='width',
        # dodge=False,
        # row='stat',
        # col='set',
        # #col='log_mult',
        # x='layer',
        # y='value',
        # #ax=axes[0],
        # #kind='line',
        # #ylabel='%',
        # #ci=100,
        # #col_wrap=2,
        # #facet_kws={
        # #    'sharey': False,
        # #    'sharex': True
        # #}
    # )
    # bp.axes[0,0].plot(np.linspace(0, N_L, num=N_L)-1, (N_L)*[errors["train"][0].iloc[0].values], color="red")

    # bp.axes[0,1].plot(np.linspace(0, N_L, num=N_L)-1, (N_L)*[errors["test"][0].iloc[0].values], color="red")

    # bp.axes[0,0].set_title("Error")
    # bp.axes[0,0].set_ylabel("error (%)")

    # # bp2 = sns.boxplot(
        # # data = df_plot.query('layer >0 & stat =="loss"'),
        # # x="layer",
        # # hue="width",
        # # doge=False,
        # # y="value",
        # # ax=axes[1]
    # # )



    # bp.axes[1,0].plot(np.linspace(0, N_L, num=N_L)-1, (N_L)*[losses["train"][0].iloc[0].values], color="red", label="full network")
    # bp.axes[1,1].plot(np.linspace(0, N_L, num=N_L)-1, (N_L)*[losses["test"][0].iloc[0].values], color="red", label="full network")
    # bp.axes[1,0].set_title("Loss")
    # bp.axes[1,0].set_ylabel("loss")
    # #plt.legend()
    # #f.legend()
    # bp.fig.subplots_adjust(top=0.85, left=0.10)
    # plt.savefig(fname=os.path.join(output_root, 'boxplot.pdf'))

    # rp = sns.relplot(
        # data = df_plot.query('layer > 0'),
        # #col='log_mult',
        # hue='width',
        # col='set',
        # row='stat',
        # # row='stat',
        # #col='log_mult',
        # x='layer',
        # y='value',
        # #style='event',
        # markers=True,
        # #ax=axes[0],
        # kind='line',
        # legend="auto",
        # #"full",
        # #ylabel='%',
        # #ci=100,
        # #col_wrap=2,
        # facet_kws={
            # 'sharey': False,
            # 'sharex': True,
            # 'legend_out':True,
        # }
    # )


    # rp.axes[0,0].plot(np.linspace(0, N_L, num=N_L)-1, (N_L)*[errors["train"][0].iloc[0].values], color="red")

    # rp.axes[0,1].plot(np.linspace(0, N_L, num=N_L)-1, (N_L)*[errors["test"][0].iloc[0].values], color="red")

    # rp.axes[0,0].set_title("Error")
    # rp.axes[0,0].set_ylabel("error (%)")

    # # rp.axes[1,0].plot(np.linspace(0, N_L, num=N_L)-1, (N_L)*[losses["train"][0].iloc[0].values], color="red", label="full network")
    # # rp.axes[1,1].plot(np.linspace(0, N_L, num=N_L)-1, (N_L)*[losses["test"][0].iloc[0].values], color="red", label="full network")
    # rp.axes[1,0].set_title("Loss")
    # rp.axes[1,0].set_ylabel("loss")

    # #rp.axes[0,1].legend()
    # #plt.legend()
    # rp.fig.legend()
    # #rp.fig.subplots_adjust(top=0.9, left=1/rp.axes.shape[1] * 0.1)
    # rp.fig.subplots_adjust(top=0.85, left=0.10)
    # if args_model is not None and args is not None:
       # removed = "width / {}".format(args.fraction) if hasattr(args, 'fraction') and args.fraction is not None else args.remove
       # rp.fig.suptitle('ds = {}, width = {}, removed = {}, draw = {}'.format(args_model.dataset, args_model.width, removed, args.ndraw))
    # #rp.set(yscale='log')
    # #rp.set(ylabel='%')
    # plt.savefig(fname=os.path.join(output_root, 'relplot.pdf'))

    # rel_error = pd.DataFrame()
    # rel_losses = pd.DataFrame()
    # for W in quant.columns.levels[2]:  # for each width
        # idx_col = (errors.columns.get_level_values("layer") > 0) & (errors.columns.get_level_values("width") == W)
        # rel_error = pd.concat([rel_error, abs(errors.loc[:, idx_col] - errors[0][W][1]) / errors[0][W][1]], axis=1, ignore_index=False)
        # rel_losses = pd.concat([rel_losses,  abs(losses.loc[:, idx_col] - losses[0][W][1]) / losses[0][W][1]], axis=1, ignore_index=False)

    # #rel_error_plot = pd.melt(rel_error.reset_index(), id_vars="draw")#, id_vars="draw")
    # #rel_losses_plot = pd.melt(rel_losses.min(axis=0).reset_index(), id_vars="layer")#, id_vars="draw")

    df_plot = pd.melt(df_reset, id_vars='draw')#.query("layer>0")
    #errors_plot = pd.melt(errors.reset_index(), id_vars="draw").query("layer>0")
    #losses_plot = pd.melt(losses.reset_index(), id_vars="draw").query("layer>0")
    cols = ["stat", "set", "layer", "width"]
    # plt.figure()

    # if N_L == 1:
        # col = "stat"
        # col_order = ["loss", "error"]
        # row="layer"
        # row_order =[1]
    # else:
        # col = "layer"
        # col_order=range(1, N_L+1)
        # row ="stat"
        # row_order = ["loss", "error"]

    # #lp = rel_losses.min(axis=0).plot(kind='line', hue='width', x='layer')
    # mp = sns.relplot(
        # #data=rel_losses.min(axis=0).to_frame(name="loss"),
        # # data=df_plot_rel, #df_plot.pivot(index="draw", columns=cols).min(axis=0).to_frame(name="value"),
        # data=df_plot.pivot(index="draw", columns=cols).min(axis=0).to_frame(name="value"),
        # # style="layer",
        # row=row,
        # row_order = row_order,
        # #row="stat",
        # #col_order=["train", "test"],
        # col=col,
        # col_order=col_order,
        # x="width",
        # y="value",
        # kind='line',
        # legend="full",
        # # legend_out=True,
        # palette=palette,
        # hue='set',
        # hue_order=["train", "test"],
        # # style_order=["],
        # markers=True,
        # facet_kws={
            # 'legend_out': True,
            # 'sharey': 'row' if (N_L>1) else False ,
            # 'sharex': True
        # }
        # #y="value",
    # )

    # # mp.fig.set_size_inches(10, 10)
    # if args_model is not None:
        # mp.fig.suptitle("(B) FCN {}".format(args_model.dataset.upper()))

    # mp.legend.set_title("Datasets")
    # fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=False)

    # xlabels=[str(i) for i in range(N_W)]
    is_vgg=False
    dataset="MNIST"
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
            df_plot= quant.loc[:, Idx[stat, setn, 1:, :]].min(axis=0).to_frame(name="value")
            lp = sns.lineplot(
                #data=rel_losses.min(axis=0).to_frame(name="loss"),
                # data=df_plot_rel if not is_vgg else df_plot_rel.pivot(index="draw", columns=col_order).min(axis=0).to_frame(name="value"),
                data=df_plot,
                #hue="width",
                hue="layer",
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
            ax.set_title("{} {}{}".format(setn.title()+(setn=="train")*"ing", stat.title(), " (%)" if stat=="error" else ''))
            # ylabel = stat if stat == "loss" else "error (%)"


            # ax.set_xlabel("width")
            ax.tick_params(labelbottom=True)
            df_ref = quant_ref[stat,setn].to_frame(name="value")

            sns.lineplot(data=df_ref.reset_index(),
                         ax=ax,
                         # hue='layer',
                         # hue_order=["train", "test"],
                         # alpha=0.5,
                         x="width",
                         y="value",
                         legend=False,
                         )
            # ax.plot(df_ref, c='g', ls=':')
            ax.set_ylabel(None)

            for l in ax.lines[-1:]:
                l.set_linestyle(':')
                l.set_color('g')

            if k == 1:
                fig.legend(handles=ax.lines, labels=["1", "2", "Ref."], title="Layer", bbox_to_anchor=(0.9,0.8), borderaxespad=0.)
            k+=1

            plt.savefig(fname=os.path.join(output_root, f"{setn}_{stat}.pdf"), bbox_inches='tight')
    # fig.subplots_adjust(top=0.85)
    # fig.legend(ax.lines, labels=["A", "B", "Reference"], title="Experiment", loc="center right")

    # palette=sns.color_palette(n_colors=2)  # the two experiments
    fig, axes = plt.subplots(1, 1, figsize=(4, 4))

    # xlabels=[str(i) for i in range(N_W)]
    is_vgg=False
    dataset="MNIST"
    # fig.suptitle("{} {}".format('VGG' if is_vgg else 'FCN', dataset.upper()))
    k = 0

    #rp.set_axis_labels("layer", "Loss", labelpad=10)
    #quant.loc[1, Idx["loss", :, 0]].lineplot(x="layer_ids", y="value", hue="")
    for i, stat in enumerate(["error" ]):
        for j, setn in enumerate(["train"]):#, "test"]):
            if stat == "loss" and setn=="test":
                continue
            # axes[k] = rp.axes[j,i]
            # ax = axes[k]
            ax = axes

            # df_plot = quant.loc[:, Idx[:, stat, setn, :]].min(axis=0).to_frame(name="value")
            df_plot= quant.loc[:, Idx[stat, setn, 1:, :]].min(axis=0).to_frame(name="value")
            lp = sns.lineplot(
                #data=rel_losses.min(axis=0).to_frame(name="loss"),
                # data=df_plot_rel if not is_vgg else df_plot_rel.pivot(index="draw", columns=col_order).min(axis=0).to_frame(name="value"),
                data=df_plot,
                #hue="width",
                hue="layer",
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
            df_ref = quant_ref[stat,setn].to_frame(name="value")

            sns.lineplot(data=df_ref.reset_index(),
                         ax=ax,
                         # hue='layer',
                         # hue_order=["train", "test"],
                         # alpha=0.5,
                         x="width",
                         y="value",
                         legend=False,
                         )
            # ax.plot(df_ref, c='g', ls=':')
            ax.set_ylabel(None)

            for l in ax.lines[-1:]:
                l.set_linestyle(':')
                l.set_color('g')

            k+=1

    # fig.subplots_adjust(top=0.85)
    fig.legend(handles=ax.lines, labels=["1", "2", "Ref."], title="Layer", bbox_to_anchor=(0.9,0.8), borderaxespad=0.)#, bbox_transform=fig.transFigure)
    # fig.legend(ax.lines, labels=[], title="Experiment", loc="center right")
    plt.margins()
    plt.savefig(fname=os.path.join(output_root, "error_train.pdf"), bbox_inches='tight')

    plt.close('all')
    return

def process_csv(file_csv):
    '''Read and process a previously computed result stored inside a checkpoint'''

    idx = pd.IndexSlice
    quant = pd.read_csv(file_csv, header=[0,1], index_col=0)
    process_df(quant, os.path.dirname(file_csv))
    return

if __name__ == '__main__':

    torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser('Evaluating a copy of a classifier with removed units')
    parser_device= parser.add_mutually_exclusive_group(required=False)
    parser_device.add_argument('--cpu', action='store_true', dest='cpu', help='force the cpu model')
    parser_device.add_argument('--cuda', action='store_false', dest='cpu')
    parser.add_argument('dirs', nargs='*', help='the directories to process')
    parser.add_argument('--table_format', choices=["wide", "long"], default="long")
    parser.set_defaults(cpu=False)



    args = parser.parse_args()
    table_format = args.table_format

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    #device = torch.device('cpu')


    dtype = torch.float
    num_gpus = torch.cuda.device_count()

    def get_parent(path):
        return os.path.basename(os.path.dirname(path))


    for directory in args.dirs:

        lst_file = glob.glob(os.path.join(directory, "**", "eval_copy.pth"), recursive=True)  # all the saved results
        unique_ids = set(list(map(get_parent, lst_file)))

        for uid in unique_ids:
            id_lst_file = glob.glob(os.path.join(directory, "**", uid, "eval_copy.pth"), recursive=True)
            df_bundle = pd.DataFrame()

            for f in id_lst_file:

                chkpt = torch.load(f, map_location=device)
                f_model = os.path.join(os.path.dirname(os.path.dirname(f)), "checkpoint.pth")  # the original model
                chkpt_model =torch.load(f_model, map_location=device)

                args_chkpt  = chkpt['args']
                args_model = chkpt_model['args']
                width = args_model.width
                N_L = args_model.depth
                N_T = args_chkpt.ndraws
                layers = np.arange(1, N_L+1)#classifier.n_layers)  # the different layers, forward order
                stats = ['loss', 'error']
                sets = ['train', 'test']


                Idx = pd.IndexSlice

                    #index = pd.Index(np.arange(1, start_epoch+args.nepoch+1), name='epoch')


                quant = chkpt['quant'].sort_index(axis=1)
                C = len(quant.columns)
                #level_width = C*[width]
                #levels = [[width]] + list(map(list, quant.columns.levels))
                levels = list([[width]] +quant.columns.levels)
                quant.columns = pd.MultiIndex.from_product(levels,
                                                        names= ['width'] + quant.columns.names,
                                                        )

                df_bundle = pd.concat([df_bundle, quant], ignore_index=False, axis=1)


            df_bundle = df_bundle.sort_index(level=0, axis=1)
            process_df(df_bundle, directory, uid, args_model=args_model)

    sys.exit(0)





