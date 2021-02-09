import torch
import numpy as np
import pandas as pd
import os
import sys
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import seaborn as sns
sns.set_theme()


import models

import torch.optim
import torch
import argparse
import utils



if __name__ == '__main__':

    torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser('Training a classifier to inspect the layers')
    parser.add_argument('--name', default='check_seq', type=str, help='the name of the experiment')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3, help='manual learning rate')
    parser.add_argument('--momentum', type=float, default=0.95, help="the momentum for SGD")
    parser.add_argument('--lr_update', '-lru', type=int, default=0, help='if any, the update of the learning rate')
    parser.add_argument('--lr_mode', '-lrm', default="manual", choices=["max", "hessian", "num_param_tot", "num_param_train", "manual"], help="the mode of learning rate attribution")
    parser.add_argument('--lr_step', '-lrs', type=int, default=30, help='if any, the step for the learning rate scheduler')
    parser.add_argument('--lr_gamma',  type=float, default=0.5, help='the gamma mult factor for the lr scheduler')
    parser.add_argument('--save_model', action='store_true', default=True, help='stores the model after some epochs')
    parser.add_argument('--nepoch', type=int, default=400, help='the number of epochs to train for')
    parser.add_argument('--batch_size', '-bs', type=int, default=100, help='the dimension of the batch')
    parser.add_argument('--debug', action='store_true', help='debug')
    parser.add_argument('--ndraw', type=int, default=20, help='The number of permutations to test')
    parser.add_argument('-F', '--fraction', type=int, default=2, help='the denominator of the removed fraction of the width')
    parser_model = parser.add_mutually_exclusive_group(required=True)
    parser_model.add_argument('--model', help='path of the model to separate')
    parser_model.add_argument('--checkpoint', help='path of the previous computation checkpoint')
    parser_device = parser.add_mutually_exclusive_group()
    parser_device.add_argument('--cpu', action='store_true', dest='cpu', help='force the cpu model')
    parser_device.add_argument('--cuda', action='store_false', dest='cpu')
    parser.add_argument('--entry_layer', type=int, default=1, help='the layer ID for the tunnel entry')
    #parser.add_argument('--end_layer', type=int, help='if set the maximum layer for which to compute the separation (forward indexing)')
    parser.set_defaults(cpu=False)



    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    #device = torch.device('cpu')


    dtype = torch.float
    num_gpus = torch.cuda.device_count()

    if args.checkpoint is not None:  # continuing previous computation
        try:
            nepoch = args.nepoch
            checkpoint = torch.load(args.checkpoint, map_location=device)
            args.__dict__.update(checkpoint['args'].__dict__)
            args.nepoch = nepoch
            cont = True  # continue the computation
        except RuntimeError:
            print('Could not load the model')


    else:
        checkpoint = dict()

    try:
        checkpoint_model = torch.load(args.model, map_location=device)  # checkpoint is a dictionnary with different keys
        root_model = os.path.dirname(args.model)
    except RuntimeError as e:
        print('Error loading the model at {}'.format(e))




    if args.entry_layer == 0:
        # no need to draw in this case
        args.ndraw = 1

    args_model = checkpoint_model['args']  # restore the previous arguments


    path_output = os.path.join(root_model, args.name)
    # Logs
    log_model = os.path.join(root_model, 'logs.txt')
    str_entry = 'entry_{}'.format(args.entry_layer)
    #draw_idx = utils.find_draw_idx(path_output)


    os.makedirs(path_output, exist_ok=True)

    if not args.debug:
        logs = open(os.path.join(path_output, 'logs_entry_{}.txt'.format(args.entry_layer)), 'w')
    else:
        logs = sys.stdout
#     logs = None

    print(os.sep.join((os.path.abspath(__file__).split(os.sep)[-2:])), file=logs)  # folder + name of the script
    print('device= {}, num of gpus= {}'.format(device, num_gpus), file=logs)
    print('dtype= {}'.format(dtype), file=logs)

    for k, v in vars(args).items():
        print("%s= %s" % (k, v), file=logs, flush=True)


    #imresize = (256, 256)
    #imresize=(64,64)
    transform =utils.parse_transform(log_model)
    train_dataset,  test_dataset, num_chs = utils.get_dataset(dataset=args_model.dataset,
                                                          dataroot=args_model.dataroot,
                                                              tfm=transform,
                                                            )
    print('Transform: {}'.format(train_dataset.transform), file=logs, flush=True)
    train_loader, size_train,\
    test_loader, size_test  = utils.get_dataloader( train_dataset,
                                                    test_dataset, batch_size
                                                    =args.batch_size,
                                                    collate_fn=None,
                                                    pin_memory=False)

    #model = models.cnn.CNN(1)

    imsize = next(iter(train_loader))[0].size()[1:]
    input_dim = imsize[0]*imsize[1]*imsize[2]



    archi = utils.parse_archi(log_model)
    model = utils.construct_FCN(archi)
    try:
        model.load_state_dict(checkpoint_model['model'])
        model.requires_grad_(False)
        model.eval()
    except RuntimeError as e:
        print("Can't load mode (error {})".format(e))

    #classifier = models.classifiers.Linear(model, args.ndraw, args.keep_ratio).to(device)
    #classifier = models.classifiers.ClassifierFCN(model, num_tries=args.ndraw, Rs=args.remove, depth_max=args.depth_max).to(device)
    remove = 1/args.fraction

    classifier = models.classifiers.AnnexFCN(model,
                                                        num_tries=args.ndraw,
                                                        R=remove,
                                                        depth=args.entry_layer).to(device)


    if 'classifier' in checkpoint.keys():
        classifier.load_state_dict(checkpoint['classifier'])

    num_parameters = utils.num_parameters(classifier, only_require_grad=False)
    num_parameters_trainable = utils.num_parameters(classifier, only_require_grad=True)
    num_layers = 1
    num_samples_train = size_train
    num_samples_test = size_test
    print('Number of parameters: {}'.format(num_parameters), file=logs)
    print('Number of trainable parameters: {}'.format(num_parameters_trainable), file=logs)
    print('Number of training samples: {}'.format(num_samples_train), file=logs)
    print('Number of testing samples: {}'.format(size_test), file=logs)
    #print('Layer dimensions'.format(classifier.size_out), file=logs)
    print('Image dimension: {}'.format(imsize), file=logs)



    print('Annex classifier: {}'.format(str(classifier)), file=logs)

    #optimizer = torch.optim.AdamW(
    #        )
    #optimizer = torch.optim.RMSprop(parameters, lr=args.learning_rate)

    def zero_one_loss(x, targets):
        ''' x: TxBxC
        targets: Bx1

        returns: error of size T
        '''
        return  (x.argmax(dim=-1)!=targets).float().mean(dim=-1)

    #mse_loss = nn.MSELoss()
    if args.ndraw == 1 or args.entry_layer == 0:  # only one try
        ce_loss = nn.CrossEntropyLoss(reduction='none')
    else:
        def ce_loss(input, target):
            '''Batch cross entropy loss

            input: TxBxC output of the linear model
            target: Bx1: the target classes

            output: TxB the loss for each try
            '''


            T, B, C = input.size()
            cond = input.gather(2,target.view(1, -1, 1).expand(T, -1, -1)).squeeze(2)  # TxBx1
            # else:
               # B, C = input.size()
               # cond = input.gather(1, target.view(-1, 1)).squeeze()
            output = - cond + input.logsumexp(dim=-1)
            return output

    learning_rate = args.learning_rate
    print('Learning rate: {}'.format(learning_rate), file=logs, flush=True)
    parameters = [p for p in classifier.network.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(parameters, momentum=args.momentum, lr=learning_rate, nesterov=True,)

    lr_scheduler = None
    if args.lr_step>0:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    print('Optimizer: {}'.format(optimizer), file=logs, flush=True)
    if 'lr_scheduler' in checkpoint.keys() and checkpoint['lr_scheduler'] is not None:

        try:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        except RuntimeError as e:
            print("Can't load model (error {})".format(e))
    #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    # divdes by 10 after the first epoch
    #lr_lambdas = [lambda epoch: (epoch == 1) * 1  + (epoch > 1)*1 for _ in param_list]
    #lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99, lr_min=1e-3)

    if 'optimizer' in checkpoint.keys():
        optimizer.load_state_dict(checkpoint['optimizer'])

    if 'lr_scheduler' in checkpoint.keys():
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    start_epoch = checkpoint.get('epochs', 0)

    sets = ['train', 'test']
    stats = ['loss', 'error']
    #layers = np.arange(1, 1+1)#classifier.n_layers)  # the different layers, forward order
    tries = np.arange(1, 1+args.ndraw)  # the different tries

    names=['set', 'stat', 'draw']
    columns=pd.MultiIndex.from_product([sets, stats, tries], names=names)
    index = pd.Index(np.arange(1, start_epoch+args.nepoch+1), name='epoch')
    quant = pd.DataFrame(columns=columns, index=index, dtype=float)

    quant.sort_index(axis=1, inplace=True)  # sort for quicker access

    if 'quant' in checkpoint.keys():
        quant.update(checkpoint['quant'])




    def get_checkpoint():
        '''Get current checkpoint'''
        global model, stats, quant, args, args_model, optimizer, lr_scheduler, epoch#, params_discarded, end

        #optimizer.param_groups = optimizer.param_groups + params_discarded

        checkpoint = {
                'classifier': classifier.state_dict(),
                'stats': stats,
            'quant': quant,
                'args': args,
            'args_model': args_model,
                'optimizer': optimizer.state_dict(),
            'lr_scheduler':lr_scheduler.state_dict() if lr_scheduler is not None else None,
                'epochs': epoch,
                    }

        #optimizer.param_groups = optimizer.param_groups[:end]

        return checkpoint

    def save_checkpoint(checkpoint=None, name=None, fname=None):
        '''Save checkpoint to disk'''

        global path_output, args

        if name is None:
            name = "checkpoint_entry_{}".format(args.entry_layer)

        if fname is None:
            fname = os.path.join(path_output, name + '.pth')

        if checkpoint is None:
            checkpoint = get_checkpoint()

        torch.save(checkpoint, fname)



    def eval_epoch(model, dataloader):
        """Evaluate the model on the dataloader"""

        global ce_loss, zero_one_loss

        model.eval()
        #loss_hidden_tot = np.zeros(classifier.L)  # for the
        err_mean = np.zeros(args.ndraw)
        loss_mean = np.zeros(args.ndraw)
        #ones_hidden = torch.ones(classifier.L, device=device, dtype=dtype)

        with torch.no_grad():
            for idx, (x, y) in enumerate(dataloader):

                x = x.to(device)
                y = y.to(device)
                out_class = model(x)  # BxC,  # each output for each layer
                loss = ce_loss(out_class, y)  # LxTxB

                x = x.to(device)
                y = y.to(device)
                out = classifier(x)  # TxBxC, LxBxC  # each output for each layer
                loss = ce_loss(out, y)  # LxTxB
                err_mean += zero_one_loss(out, y).detach().cpu().numpy()
                loss_mean = (idx * loss_mean + loss.mean(dim=-1).detach().cpu().numpy())/(idx+1)


        return loss_mean, err_mean/idx




    stop = False
    separated = False
    epoch =  start_epoch
    frozen = False
    ones = torch.ones(args.ndraw, device=device, dtype=dtype) if args.ndraw > 1 else torch.ones([])




    while not stop:
    #for epoch in tqdm(range(start_epoch, start_epoch+args.nepoch)):


        classifier.train()
        #loss_hidden_tot = np.zeros(classifier.L)  # for the
        loss_train = np.zeros(args.ndraw)  # for the
        err_train = np.zeros(args.ndraw)
        #ones_hidden = torch.ones(classifier.L, device=device, dtype=dtype)

        for idx, (x, y) in enumerate(train_loader):


            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out_class = classifier(x)  # TxBxC,  # each output for each layer
            loss = ce_loss(out_class, y)  # LxTxB
            #loss_hidden = ce_loss(out_hidden, y)
            error = zero_one_loss(out_class, y)  # T
            err_train = (idx * err_train + error.detach().cpu().numpy()) / (idx+1)
            loss_train = (idx * loss_train + loss.mean(dim=-1).detach().cpu().numpy()) / (idx+1)
        # loss_hidden_tot = (idx * loss_hidden_tot + loss_hidden.mean(dim=1).detach().cpu().numpy()) / (idx+1)
            if not frozen:  # if we have to update the weights
                loss.mean(dim=-1).backward(ones)
            # loss_hidden.mean(dim=1).backward(ones_hidden)
                optimizer.step()
                #lr_scheduler.step()

        epoch += 1 if not frozen else 0

        err_min = err_train.min(axis=0)  # min over tries


        separated = frozen and err_min == 0
        frozen = err_min == 0 and not frozen # will test with frozen network next time, prevent from freezing twice in a row

        if frozen:
            print("Freezing the next iteration", file=logs)

        stop = (separated
                or epoch > start_epoch + args.nepoch
                )


        quant.loc[pd.IndexSlice[epoch, ('train', 'error')]] =  err_train.reshape(-1)
        quant.loc[pd.IndexSlice[epoch, ('train', 'loss')]] =  loss_train.reshape(-1)


        loss_test, err_test = eval_epoch(classifier, test_loader)

        quant.loc[pd.IndexSlice[epoch, ('test', 'error')]] =  (err_test).reshape(-1)
        quant.loc[pd.IndexSlice[epoch, ('test', 'loss')]] =  loss_test.reshape(-1)



        #print('ep {}, train loss (error) {:g} ({:g}), test loss (error) {:g} ({:g})'.format(
        print('ep {}, train loss (min/max): {:g} / {:g}, error (min/max): {:g}/{:g} {}'.format(
            epoch, quant.loc[epoch, ('train', 'loss')].min(), quant.loc[epoch, ('train', 'loss')].max(),
            err_min, quant.loc[epoch, ('train', 'error')].max(), ' (frozen)' if frozen else ''),
            file=logs, flush=True)

        if args.lr_step>0:
            lr_scheduler.step()
        #end_layer = 1
        if epoch % 5 == 0 or stop:

            quant_reset = quant.reset_index()
            quant_plot = pd.melt(quant_reset, id_vars='epoch')
            g = sns.relplot(
                data = quant_plot,
                #col='layer',
                hue='set',
                row='stat',
                x='epoch',
                y='value',
                kind='line',
                ci=100,  # the whole spectrum of the data
                facet_kws={
                'sharey': 'row',
                'sharex': True
            }
            )

            g.set(yscale='log')
            #g.set(title='ds = {}, width = {}, removed = {}, Tries = {}'.format(args_model.dataset, args_model.width, args.remove, args.ndraw))
            g.fig.subplots_adjust(top=0.9, left=1/g.axes.shape[1] * 0.1 )  # number of columns in the sublplot
            try:
                g.fig.suptitle('ds = {}, width = {}, removed = {}, Tries = {}, name = {}'.format(args_model.dataset, args_model.width, args.remove, args.ndraw, args.name))
            except:
                pass
            #g.set_axis_labels

            plt.savefig(fname=os.path.join(path_output, 'losses_entry_{}.pdf'.format(args.entry_layer)), bbox_inches="tight")

            plt.close('all')

            if args.save_model:
                save_checkpoint()


    if separated:
        print("Data is separated.", file=logs)
        sys.exit(0)  # success
    else:
        print("Data is NOT separated.", file=logs)
        sys.exit(1)  # failure



