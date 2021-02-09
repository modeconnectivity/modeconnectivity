import torch
import numpy as np
import pandas as pd
import os
import sys
import torch.nn as nn
from collections import OrderedDict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import seaborn as sns
sns.set(font_scale=3, rc={'text.usetex' : False})
sns.set_theme()
sns.set_style('whitegrid')
import glob
from scipy.optimize import minimize_scalar


import models

import torch.optim
import torch
import argparse
import utils




def process_df(quant, dirname, is_vgg=False, args=None, args_model=None, save=True):

    global table_format
    idx = pd.IndexSlice
    n = len(quant.columns.levels)
    #losses = quant.xs('loss', level=n-1, axis=1)
    #cols_error = idx[:, :, 'error'] if n == 3 else idx[:, 'error']
    col_order = ["stat", "set", "layer"]
    quant = utils.assert_col_order(quant, col_order, id_vars="draw", values="value")
    # if quant.columns.names != col_order:

    cols_error = idx['error', :, :]
    #quant.loc[:, cols_error] *= 100  # in %
    quant = quant.sort_index(axis=1)
    N_L = len(quant.columns.unique(level="layer")) -1 # number of hidden layers
    errors = quant["error"]
    losses = quant["loss"]
    #errors = quant["error"]


    if save:
        quant.to_csv(os.path.join(dirname, 'quant.csv'))

    quant["error"] *= 100

    df_reset = quant.reset_index()
    df_plot = pd.melt(df_reset, id_vars='draw')#.query("layer>0")
    df_plot_no_0 = df_plot.query('layer>0')
    df_plot_0 = df_plot.query('layer==0')
    #relative quantities
    quant_ref = quant.loc[1, Idx[:, :, 0]]
    N_S = len(quant_ref)
    quant_ref_val = quant_ref.iloc[np.repeat(np.arange(N_S), N_L)].values
    quant_rel = (quant.loc[:, Idx[:, :, 1:]] - quant_ref_val).abs()
    df_reset_rel = quant_rel.reset_index()
    df_plot_rel = pd.melt(df_reset_rel, id_vars="draw")

    palette=sns.color_palette(n_colors=2)


    if N_L == 0:
        sys.exit(0)



    rp = sns.relplot(
        #data=rel_losses.min(axis=0).to_frame(name="loss"),
        data=df_plot_rel if not is_vgg else df_plot_rel.pivot(index="draw", columns=col_order).min(axis=0).to_frame(name="value"),
        #hue="width",
        hue="set",
        hue_order=["train", "test"],
        col="stat",
        col_order=["loss", "error"],
        x="layer",
        y="value",
        kind='line',
        legend="full",
        style='set',
        ci='sd',
        palette=palette,
        #style='layer',
        markers=False,
        dashes=False,
        #legend_out=True,
        facet_kws={
            'sharey': False,
            'sharex': True
        }
        #y="value",
    )

    if not is_vgg:
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
            style='set',
            legend=False,
            ax=rp.axes[0,0],
            alpha=0.5,
            #palette=sns.color_palette(n_colors=N_L),
            #style='layer',
            markers=False,
            dashes=True,
            #legend_out=True,
            # facet_kws={
                # 'sharey': False,
                # 'sharex': True
            # }
            #y="value",
        )
        #plt.figure()
        for ax in rp.axes[0,0].lines[-2:]:  # the last two
            ax.set_linestyle('--')


        sns.lineplot(
            #data=rel_losses.min(axis=0).to_frame(name="loss"),
            data=df_plot_rel.query("stat=='error'").pivot(index="draw", columns=col_order).abs().min(axis=0).to_frame(name="value"),
            #hue="width",
            hue="set",
            hue_order=["train", "test"],
            #col="stat",
            #col_order=["loss", "error"],
            x="layer",
            y="value",
            #kind='line',
            #legend="full",
            style='set',
            ax=rp.axes[0,1],
            alpha=0.5,
            #palette=sns.color_palette(n_colors=N_L),
            #style='layer',
            legend=False,
            markers=False,
            dashes=True,
            #legend_out=True,
            # facet_kws={
                # 'sharey': False,
                # 'sharex': True
            # }
            #y="value",
        )

        for ax in rp.axes[0,1].lines[-2:]:  # the last two
            ax.set_linestyle('--')

    if args_model is not None:
        rp.fig.suptitle("(B) {} {}".format('VGG' if is_vgg else 'FCN', args_model.dataset.upper()))

    rp.axes[0,0].set_title("Loss")
    rp.axes[0,0].set_ylabel("absolute delta loss")
    rp.axes[0,0].set_xlabel("layer index l")
    #rp.set_axis_labels("layer", "Loss", labelpad=10)
    #quant.loc[1, Idx["loss", :, 0]].lineplot(x="layer_ids", y="value", hue="")
    # sns.lineplot(
        # data=df_plot_0.query('stat=="loss"').dropna().iloc[np.tile(np.arange(2), N_L)].reset_index(),  # repeat the datasaet N_L times
        # hue='set',
        # hue_order=["train", "test"],
        # ax=rp.axes[0,0],
        # x=np.tile(np.linspace(1, N_L, num=N_L), 2),
        # style='set',
        # dashes=True,
        # legend=False,
        # y="value",)


    rp.axes[0,1].set_ylabel("absolute delta error (%)")
    # #rp.axes[0,1].set_ylabel("error (%)")
    rp.axes[0,1].set_title("Error")
    rp.axes[0,1].set_xlabel("layer index l")


    #    ax.legend()
    plt.savefig(fname=os.path.join(dirname, "rel_quant.pdf"), bbox_inches='tight')

    #mp.axes[1,1].set_ylabel("loss")

    rp = sns.relplot(
        #data=rel_losses.min(axis=0).to_frame(name="loss"),
        data=df_plot_no_0.pivot(index="draw", columns=col_order).abs().min(axis=0).to_frame(name="value"),
        #hue="width",
        hue="set",
        hue_order=["train", "test"],
        col="stat",
        col_order=["loss", "error"],
        x="layer",
        y="value",
        kind='line',
        legend="full",
        style='set',
        #palette=sns.color_palette(n_colors=N_L),
        #style='layer',
        markers=True,
        dashes=False,
        #legend_out=True,
        facet_kws={
            'sharey': False,
            'sharex': True
        }
        #y="value",
    )

    #plt.figure()
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
    parser.add_argument('--name', default='B', type=str, help='the name of the experiment')
    parser_model = parser.add_mutually_exclusive_group(required=True)
    parser_model.add_argument('--model', nargs='*', help='path of the model to separate')
    parser_model.add_argument('--root_model', nargs='*', help='path of the model to separate')
    parser_normalized = parser.add_mutually_exclusive_group()
    parser_normalized.add_argument('--normalized', action='store_true', dest='normalized',  help='normalized the input')
    parser_normalized.add_argument('--no-normalized', action='store_false', dest='normalized', help='normalized the input')
    parser.set_defaults(normalized=False)
    parser_device = parser.add_mutually_exclusive_group()
    parser_device.add_argument('--cpu', action='store_true', dest='cpu', help='force the cpu model')
    parser_device.add_argument('--cuda', action='store_false', dest='cpu')
    parser.add_argument('--ndraw', type=int, default=20, help='The number of ndraw to take')
    parser.add_argument('--table_format', choices=["wide", "long"], default="long")
    parser.add_argument('--fraction', '-F', default=2, type=int, help='the removed neurons denominator')
    parser.set_defaults(cpu=False)



    args = parser.parse_args()
    table_format = args.table_format

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    #device = torch.device('cpu')


    dtype = torch.float
    num_gpus = torch.cuda.device_count()

    if args.root_model is not None:

        lst_models = [glob.glob(os.path.join(rm, '**', 'checkpoint.pth'), recursive=True) for rm in args.root_model]
    elif args.model is not None:
        lst_models = [args.model]
    else:
        raise NotImplementedError

    for m in [m for lst in lst_models for m in lst]:
        checkpoint = dict()

        #if os.path.isfile():

        try:
            checkpoint_model = torch.load(m, map_location=device)  # checkpoint is a dictionnary with different keys
            root_model = os.path.dirname(m)
        except RuntimeError as e:
            print('Error loading the model at {}'.format(e))
        args_model = checkpoint_model['args']  # restore the previous arguments
        #imresize = checkpoint_model.get('imresize', None)
        log_model = os.path.join(os.path.dirname(m), 'logs.txt')

        path_output = os.path.join(root_model, args.name)
        os.makedirs(path_output, exist_ok=True)

        if hasattr(args_model, 'model') and args_model.model.find('vgg') != -1:
            # VGG model
            is_vgg=True
            NUM_CLASSES = utils.get_num_classes(args_model.dataset)
            model, _ = models.pretrained.initialize_model(args_model.model,
                                                pretrained=False,
                                                feature_extract=False,
                                                num_classes=NUM_CLASSES)
            model.n_layers = utils.count_hidden_layers(model)
            PrunedClassifier = models.classifiers.PrunedCopyVGG
            args.normalized=True


        else:
            is_vgg=False
            archi = utils.parse_archi(log_model)
            model = utils.construct_FCN(archi)
            PrunedClassifier = models.classifiers.PrunedCopyFCN


        transform= utils.parse_transform(log_model)
        logs = open(os.path.join(path_output, 'logs_eval.txt'), 'w')
    # Logs
        train_dataset, test_dataset, num_chs = utils.get_dataset(dataset=args_model.dataset,
                                                            dataroot=args_model.dataroot,
                                                                                tfm=transform,
                                                                                normalize=args.normalized,
                                                                                #augment=False,
                                                                #imresize =imresize,
                                                                )
        print('Transform: {}'.format(train_dataset.transform), file=logs, flush=True)
        train_loader, size_train,\
            test_loader, size_test  = utils.get_dataloader( train_dataset,
                                                            test_dataset,
                                                            batch_size =args_model.batch_size,
                                                            collate_fn=None,
                                                            pin_memory=True,
                                                            num_workers=4)
        try:
            new_keys = map(lambda x:x.replace('module.', ''), checkpoint_model['model'].keys())
            checkpoint_model['model'] = OrderedDict(zip(list(new_keys), checkpoint_model['model'].values()))
            #for key in state_dict.keys():
                #new_key = key.replace('module.', '')

            model.load_state_dict(checkpoint_model['model'])
        except RuntimeError as e:
            print("Can't load mode (error {})".format(e))
        # else: # not a file, should be a vgg name

            # checkpoint = dict()

            # dataset = args.dataset
    keep = 1 - 1 / args.fraction

#     logs = None

    print(os.sep.join((os.path.abspath(__file__).split(os.sep)[-2:])), file=logs)  # folder + name of the script
    print('device= {}, num of gpus= {}'.format(device, num_gpus), file=logs)
    print('dtype= {}'.format(dtype), file=logs)
    print('Model: {}'.format(str(model)), file=logs)

    for k, v in vars(args).items():
        print("%s= %s" % (k, v), file=logs, flush=True)




    model.requires_grad_(False)
    model.eval()

    model.to(device)

    num_samples_train = size_train
    num_samples_test = size_test
    #print('Number of parameters: {}'.format(num_parameters), file=logs)
    print('Number of training samples: {}'.format(num_samples_train), file=logs)
    print('Number of testing samples: {}'.format(size_test), file=logs)
    #print('Layer dimensions'.format(classifier.size_out), file=logs)
    def zero_one_loss(x, targets):
        ''' x: BxC
        targets: Bx1

        returns: error of size 1
        '''
        return  (x.argmax(dim=1)!=targets).float().mean(dim=0)

    #mse_loss = nn.MSELoss()
    #ce_loss_check = nn.CrossEntropyLoss(reduction='none')

    def ce_loss(input, targets):
        '''Batch cross entropy loss

        input: BxC output of the linear model
        targets: Bx1: the targets classes

        output: B the loss for each try
        '''


        B, C = input.size()
        cond = input.gather(1,targets.view(-1, 1)).squeeze(1)  # Bx1
        output = - cond + input.logsumexp(dim=-1)
        return output
    #lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99, lr_min=1e-3)

    #sets = ['train', 'test']
    N_L = utils.count_hidden_layers(model)
    layers = np.arange(1, N_L+1)#classifier.n_layers)  # the different layers, forward order
    #log_mult = np.arange(1, N_L+1)
    stats = ['loss', 'error']
    #tries = np.arange(1, 1+args.ndraw)  # the different tries

    names=['layer', 'stat', 'set']
    sets = ['train', 'test']
    columns=pd.MultiIndex.from_product([layers, stats, sets], names=names)
    #index = pd.Index(np.arange(1, start_epoch+args.nepoch+1), name='epoch')
    index = pd.Index(np.arange(1, args.ndraw+1), name='draw')
    quant = pd.DataFrame(columns=columns, index=index, dtype=float)

    quant.sort_index(axis=1, inplace=True)  # sort for quicker access

    df_mult = pd.DataFrame(columns=[layers], index=index, dtype=float)

    #if 'quant' in checkpoint.keys():
    #    quant.update(checkpoint['quant'])



    #classes = torch.arange(num_classes).view(1, -1).to(device)  # the different possible classes


    def get_checkpoint():
        '''Get current checkpoint'''
        global model, stats, quant, df_mult, args, optimizer, lr_scheduler, epoch#, params_discarded, end

        #optimizer.param_groups = optimizer.param_groups + params_discarded

        checkpoint = {
                'classifier': classifier.state_dict(),
                #'stats': stats,
            'quant': quant,
            'df_mult': df_mult,
                'args': args,
            #'log_mult': args.log_mult,
        #  'args_model': args_model,
                #'optimizer': optimizer.state_dict(),
                #'epochs': epoch,
                    }

        #optimizer.param_groups = optimizer.param_groups[:end]

        return checkpoint

    def save_checkpoint(fname=None, checkpoint=None):
        '''Save checkpoint to disk'''

        global path_output

        if fname is None:
            fname = os.path.join(path_output, 'eval_copy.pth')

        if checkpoint is None:
            checkpoint = get_checkpoint()

        torch.save(checkpoint, fname)




    def eval_epoch(classifier, dataloader):


        classifier.eval()
        #loss_hidden_tot = np.zeros(classifier.L)  # for the
        loss_tot = 0
        err_tot = 0
        #ones_hidden = torch.ones(classifier.L, device=device, dtype=dtype)

        with torch.no_grad():
            for idx, (x, y) in enumerate(dataloader):


                x = x.to(device)
                y = y.to(device)
                out_class = classifier(x)  # BxC,  # each output for each layer
                loss = ce_loss(out_class, y)  # LxTxB
                error = zero_one_loss(out_class, y)  # T
                err_tot = (idx * err_tot + error.detach().cpu().numpy()) / (idx+1)  # mean error
                loss_tot = (idx * loss_tot + loss.mean(dim=-1).detach().cpu().numpy()) / (idx+1)  # mean loss
                # loss_hidden_tot = (idx * loss_hidden_tot + loss_hidden.mean(dim=1).detach().cpu().numpy()) / (idx+1)
                #break


        return loss_tot, err_tot


    loss_0, error_0 = eval_epoch(model, train_loader)  # original loss and error of the model
    loss_test, error_test = eval_epoch(model, test_loader)
    print(f'Train loss: {loss_0}, error: {error_0}', file=logs, flush=True)
    #stats_0.to_csv(os.path.join(path_output, 'original.csv'))
    Idx = pd.IndexSlice
    quant.loc[1, Idx[0, 'loss', 'train']] = loss_0
    quant.loc[1, Idx[0, 'error', 'train']] = error_0

    quant.loc[1, Idx[0, 'loss', 'test']] = loss_test
    quant.loc[1, Idx[0, 'error', 'test']] = error_test
    print(f'Test loss: {loss_test}, error: {error_test}', file=logs, flush=True)

    def get_output_class(classifier, loader):
        """Return the entire ouptut for the whole dataset (without multiplier)"""

        out = torch.empty((len(loader), loader.batch_size, classifier.n_classes))
        Y = torch.empty((len(loader), loader.batch_size), dtype=torch.long)
        for idx, (x,y) in enumerate(loader):
            x = x.to(device)
            #y = y.to(device)
            out[idx, :, : ] = classifier.forward_no_mult(x).detach().cpu()
            Y[idx, :] = y.cpu().long()
        return out, Y


    def eval_class_mult(out_class, mult, out=None):
        """Eval the output (with multiplier first)"""


        #classifier.eval()
        X, Y = out_class
        X = mult * X  # multiply the output of the classifier by mult
        shape = X[0].shape
        #loss_hidden_tot = np.zeros(classifier.L)  # for the
        loss_tot = 0
        err_tot = 0
        #ones_hidden = torch.ones(classifier.L, device=device, dtype=dtype)

        with torch.no_grad():
            for idx, (x, y) in enumerate(zip(X, Y)):
                x = x.to(device)
                y = y.to(device)
                loss = ce_loss(x, y)  # LxTxB
                loss_tot = (idx * loss_tot + loss.mean(dim=-1).detach().cpu().numpy()) / (idx+1)  # mean loss
                error = zero_one_loss(x, y)
                err_tot = (idx * err_tot + error.mean(dim=-1).detach().cpu().numpy()) / (idx+1)  # mean error
                # loss_hidden_tot = (idx * loss_hidden_tot + loss_hidden.mean(dim=1).detach().cpu().numpy()) / (idx+1)
                #break

        if out is not None:
            out['error'] = err_tot
            return loss_tot
        else:
            return loss_tot, err_tot


    #mult = 2**args.log_mult
    for t in range(1, args.ndraw+1):
        for l in range(1, N_L+1):


            def eval_mult(mult, out_class, out):
                #global out_class
                loss  = eval_class_mult(out_class, mult, out)#epoch(classifier, train_loader, with_error=False)
                return loss


            out={'error': 0}
            classifier = PrunedClassifier(model,l, keep=keep).to(device)
            classifier.requires_grad_(False)
            if is_vgg and torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!", file=logs, flush=True)
                # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
                classifier.features = nn.DataParallel(classifier.features)
            out_class = get_output_class(classifier, train_loader)


            error = out['error']
            # mult0 = 1
            # res = minimize(eval_mult, mult0, args=out_class, method='BFGS')#l, options={'disp': True})
            #res = minimize_scalar(eval_mult, args=(out_class,), bounds=(0, 2**(N_L+2-l)), method='bounded')
            res = minimize_scalar(eval_mult, args=(out_class,out), method='brent')
            #res = minimize_scalar(eval_mult, args=(out_class,), method='golden')
            print(res, file=logs)
            loss = res.fun
            mult = res.x
            #else:
            #    mult=2**(N_L+1-l) #res.multult0

            # classifier.mult = torch.tensor(mult, device=device, dtype=dtype)
            # loss, error = eval_epoch(classifier, train_loader)
            df_mult.loc[t, l] = mult


            quant.loc[pd.IndexSlice[t, (l, 'loss', 'train')]] =  loss
            quant.loc[pd.IndexSlice[t, (l, 'error', 'train')]] =  error

            loss_test, error_test = eval_epoch(classifier, test_loader)
            quant.loc[pd.IndexSlice[t, (l, 'loss', 'test')]] =  loss_test
            quant.loc[pd.IndexSlice[t, (l, 'error', 'test')]] =  error_test


            print('mult: {}, t: {}, l: {}, loss: {} (test {}) , error: {} (test {})'.format(mult, t, l, loss, loss_test, error, error_test), file=logs, flush=(l==N_L))


        if t % 20 ==0 or t==args.ndraw:
            quant = quant.sort_index(axis=1)
            df_mult = df_mult.sort_index(axis=1)
            quant.to_csv(os.path.join(path_output, 'quant.csv'))
            df_mult.to_csv(os.path.join(path_output, 'mult.csv'))
            df_mult.describe().to_csv(os.path.join(path_output, 'mult_describe.csv'))
            save_checkpoint()
            process_df(quant, path_output, is_vgg=is_vgg)



