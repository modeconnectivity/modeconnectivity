import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import argparse
sns.set(
    font_scale=1.5,
    style="whitegrid",
    rc={
    'text.usetex' : False,
        'lines.linewidth': 2
    }
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser('Plotting of a path between two solutions')
    parser_device = parser.add_mutually_exclusive_group()
    parser_device.add_argument('--cpu', action='store_true', dest='cpu', help='force the cpu model')
    parser_device.add_argument('--cuda', action='store_false', dest='cpu')
    parser.add_argument('--yscale', default="linear", choices=['log', 'linear'], help='the scale for the y axis')
    parser.add_argument('--split', action='store_true', help='split the plot into separated figures')
    parser.add_argument('dir', help='the directory where the path is recorded')
    parser.set_defaults(cpu=False)

    args = parser.parse_args()

    dirname = args.dir
    filename = os.path.join(args.dir, 'A.csv')

    df = pd.read_csv(filename, header=[0,1,2], index_col=[0,1])
    split = args.split
    palette=sns.color_palette(n_colors=1)
    orange = sns.color_palette(n_colors=2)[1]
    Idx = pd.IndexSlice
    yscale = args.yscale
    logstr = "_log" if yscale == "log" else ""
    df_ref = None
    df_ds = None
    fref = os.path.join(dirname, "ref.csv")
    fds = os.path.join(dirname, "B.csv")
    if os.path.isfile(fref):
        df_ref = pd.read_csv(fref, index_col=[0,1], header=[0])['0']
        df_ref.loc["error", :] *= 100
        df_ref_log = np.log10(df_ref)
    if os.path.isfile(fds):
        df_ds = pd.read_csv(fds, index_col=[0,1], header=[0])['0']
        df_ds.loc["error", :] *= 100
        df_ds_log = np.log10(df_ds)
    # xlabels =
    output_root = dirname
    stat_idx = df.columns.names.index("stat")
    nlevels = df.columns.nlevels
    if "err" in df.columns.get_level_values("stat"):
        new_stat_lvl = [s.replace("err", "error") for s in df.columns.get_level_values(stat_idx)]
        # new_stat.sort()
        levels = [df.columns.get_level_values(i) if i != stat_idx else new_stat_lvl for i in range(nlevels)]
        cols = pd.MultiIndex.from_arrays(levels, names=df.columns.names)
        df.columns = cols
    df.loc[:, Idx[:, "error"]] *= 100  # in %

    if not split:
        fig, axes = plt.subplots(2, 1, figsize=(4, 8), sharey=False)
    df_log = np.log10(df)

    k = 0
    for i, stat in enumerate(["loss","error" ]):
        for j, setn in enumerate(["train","test"]):
            if stat == "loss" and setn=="test":
                continue
            if stat == "error" and setn=="train":
                continue
            # axes[k] = rp.axes[j,i]
            if split:
                fig, ax = plt.subplots(1, 1, figsize=(4, 4), sharex=False)
            else:
                ax = axes.flatten()[k]

            #log_plot = yscale == "log" and setn == "train"
            log_plot = False

            if log_plot:
                df_plot = df_log
            else:
                df_plot = df
            # lp = sns.lineplot(
            df_plot.plot(kind="line",

                #data=rel_losses.min(axis=0).to_frame(name="loss"),
                # hue_order=keys,
                # x="layer",
                y=(setn,stat),
                legend=None,
                # style='set',
                # ci='sd',
                # palette=palette,
                #style='layer',
                # markers=False,
                ax=ax,
                # dashes=True,
                # linewidth=3.,
                #legend_out=True,
                #y="value",
            )
            # lp.set_xticklabels(xlabels)#, rotation=40*(is_vgg))
            # else:
                # lp.set_xticklabels(len(xlabels)*[None])

            if not split:
                ax.set_title("{} {}{}".format(setn.title()+(setn=="train")*"ing", stat.title(), " (%)" if stat=="error" else ''))
            # ylabel = stat if stat == "loss" else "error (%)"
            # ax.set_xlabel("layer index l")
            ax.set_ylabel(None)
            # ax.tick_params(labelbottom=True)


            if df_ds is not None:
                # data_ref  = quant_ref[stat, setn].reset_index()

                if log_plot:
                    ax.axline((0,df_ds_log[stat, setn]), (1, df_ds_log[stat, setn]),  zorder=2, c=orange)
                else:
                    ax.axline((0,df_ds[stat, setn]), (1, df_ds[stat, setn]),  zorder=2, c=orange)
                # data_ds.index = pd.Index(range(len(data_ds)))
                    # ax=ax,
            if df_ref is not None:
                # data_ref  = quant_ref[stat, setn].reset_index()

                if log_plot:
                    ax.axline((0,df_ref_log[stat, setn]), (1, df_ref[stat, setn]),  ls=":", zorder=2, c='g')
                else:
                    ax.axline((0,df_ref[stat, setn]), (1, df_ref[stat, setn]),  ls=":", zorder=2, c='g')
                # data_ref.index = pd.Index(range(len(data_ref)))
                    # ax=ax,

            if setn == "train":
                ax.set_yscale(yscale)
                # ax.get_yaxis().set_major_formatter(lambda x, pos: str(int(np.log10(x))))



            if split:
                # if k == 1:
                labels= ["A", "B", "ref."]
                if setn == "test":
                    logstr = ""
                fig.legend(handles=ax.lines, labels=labels,
                            # title="Exp.",
                            loc="upper right", borderaxespad=0, bbox_to_anchor=(0.9,0.8))#, bbox_transform=fig.transFigure)

                # fig.tight_layout()
                plt.margins()

                plt.savefig(fname=os.path.join(output_root, f"{setn}_{stat}{logstr}.pdf"), bbox_inches='tight')

            k += 1

    # fig.subplots_adjust(top=0.85)
    # if is_vgg:
    if not split:
        labels=["A", "B", "ref."]
        fig.legend(handles=ax.lines, labels=labels,
                  # title="Exp.",
                   loc="upper right", borderaxespad=0, bbox_to_anchor=(0.9,0.8))#, bbox_transform=fig.transFigure)
        fig.tight_layout()
        # plt.margins()
        fig.savefig(fname=os.path.join(output_root, f"train_loss_test_error{logstr}.pdf"), bbox_inches='tight')
