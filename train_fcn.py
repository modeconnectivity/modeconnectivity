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

    parser = argparse.ArgumentParser('Training a classifier to inspect the layers')
    parser.add_argument('--dataset', '-dat', default='cifar10', type=str, help='dataset')
    parser.add_argument('--dataroot', '-dr', default='./data/', help='the root for the input data')
    parser.add_argument('--output_root', '-oroot', type=str, help='output root for the results')
    parser.add_argument('--name', default = '', type=str, help='the name of the experiment')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-2, help='leraning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help="the weight decay for SGD (L2 pernalization)")
    parser.add_argument('--momentum', type=float, default=0.95, help="the momentum for SGD")
    parser.add_argument('--lr_mode', '-lrm', default="manual", choices=["max", "hessian", "num_param", "manual"], help="the mode of learning rate attribution")
    parser.add_argument('--lr_step', '-lrs', type=int, default=30, help='if any, the step for the learning rate scheduler')
    parser.add_argument('--lr_gamma',  type=float, default=0.5, help='the gamma mult factor for the lr scheduler')
    parser.add_argument('--lr_update', '-lru', type=int, default=0, help='if any, the update of the learning rate')
    parser.add_argument('--save_model', action='store_true', default=True, help='stores the model after some epochs')
    parser.add_argument('--nepoch', type=int, default=1000, help='the number of epochs to train for')
    parser.add_argument('--depth', '-L', type=int, default=3, help='the number of layers for the network')
    parser_normalize = parser.add_mutually_exclusive_group()
    parser_normalize.add_argument('--normalize', action='store_true', dest='normalize',  help='normalize the input')
    parser_normalize.add_argument('--no-normalize', action='store_false', dest='normalize', help='normalize the input')
    parser.set_defaults(normalize=True)
    parser.add_argument('--batch_size', '-bs', type=int, default=100, help='the dimension of the batch')
    parser.add_argument('--debug', action='store_true', help='debug')
    parser.add_argument('--size_max', type=int, default=None, help='maximum number of traning samples')
    parser.add_argument('--width', '-w', type=int, help='The width of the layers')
    parser.add_argument('--checkpoint', help='path of the previous computation checkpoint')
    parser.add_argument('--gd_mode', '-gdm', default='stochastic', choices=['full', 'stochastic'], help='whether the gradient is computed full batch or stochastically')


    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float
    num_gpus = torch.cuda.device_count()



    if args.checkpoint is not None:  # we have some networks weights to continue
        try:
            nepoch=args.nepoch
            checkpoint = torch.load(args.checkpoint, map_location=device)
            args.__dict__.update(checkpoint['args'].__dict__)
            args.nepoch=nepoch

        except RuntimeError as e:
            print('Error loading the checkpoint at {}'.format(e))

    else:
        checkpoint = dict()





    # Logs

    if args.output_root is None:
        # default output directory
        args.output_root = utils.get_output_root(args)





    path_output = os.path.join(args.output_root, args.name)
    #path_checkpoints = join_path(path_output, 'checkpoints')

    os.makedirs(path_output, exist_ok=True)
    #os.makedirs(path_checkpoints, exist_ok=True)


    if not args.debug:
        logs = open(os.path.join(path_output, 'logs.txt'), 'w')
    else:
        logs = sys.stdout
#     logs = None

    print(os.sep.join((os.path.abspath(__file__).split(os.sep)[-2:])), file=logs)  # folder + name of the script
    print('device= {}, num of gpus= {}'.format(device, num_gpus), file=logs)
    print('dtype= {}'.format(dtype), file=logs)

    for k, v in vars(args).items():
        print("%s= %s" % (k, v), file=logs, flush=True)


    train_dataset, test_dataset, num_chs = utils.get_dataset(dataset=args.dataset,
                                                          dataroot=args.dataroot,
                                                                normalize=True,
                                                             )
    print('Transform: {}'.format(train_dataset.transform), file=logs, flush=True)
    train_loader, size_train,\
        test_loader, size_test  = utils.get_dataloader( train_dataset,
                                                       test_dataset,
                                                       batch_size =args.batch_size,
                                                       size_max=args.size_max,
                                                       num_workers=4,
                                                       pin_memory=True,
                                                       collate_fn=None)


    num_classes = utils.get_num_classes(args.dataset)
    imsize = next(iter(train_loader))[0].size()[1:]
    input_dim = imsize[0]*imsize[1]*imsize[2]



    model = models.classifiers.FCNHelper(num_layers=args.depth,
                                         input_dim=input_dim,
                                         num_classes=num_classes,
                                         width=args.width)

    num_parameters = utils.num_parameters(model)
    num_samples_train = size_train
    num_samples_test = size_test
    print('Number of parameters: {}'.format(num_parameters), file=logs)
    print('Number of training samples: {}'.format(num_samples_train), file=logs)
    print('Number of testing samples: {}'.format(size_test), file=logs)
    print('Image size:'.format(imsize), file=logs)
    print('Model: {}'.format(str(model)), file=logs)
    model.to(device)

    if 'model' in checkpoint.keys():
        try:
            model.load_state_dict(checkpoint['model'])
            model.train()
        except RuntimeError as e:
            print("Can't load mode (error {})".format(e))

    # error
    def zero_one_loss(x, targets):
        return  (x.argmax(dim=1)!=targets).float().mean()

    # loss
    ce_loss = nn.CrossEntropyLoss()


    parameters = list(model.parameters())

    optimizer = torch.optim.SGD(
        #parameters, lr=args.learning_rate, momentum=(args.gd_mode=='full') * 0 + (args.gd_mode =='stochastic')*0.95
        parameters, lr=args.learning_rate, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay,
    )

    print("Optimizer: {}".format(optimizer), file=logs, flush=True)
    #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    lr_scheduler = None
    if args.lr_step>0:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    if 'optimizer' in checkpoint.keys():

        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except RuntimeError as e:
            print("Can't load model (error {})".format(e))

    if 'lr_scheduler' in checkpoint.keys() and checkpoint['lr_scheduler'] is not None:

        try:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        except RuntimeError as e:
            print("Can't load model (error {})".format(e))

    start_epoch = 0
    if 'epochs' in checkpoint.keys():
        start_epoch = checkpoint['epochs']

    # the output quantitites
    names=['set', 'stat']
    sets = ['train', 'test']
    stats = ['loss', 'error']

    columns=pd.MultiIndex.from_product([sets, stats], names=names)
    index = pd.Index(np.arange(1, args.nepoch+start_epoch), name='epoch')
    quant = pd.DataFrame(columns=columns, index=index, dtype=float)


    if 'quant' in checkpoint.keys():
        quant.update(checkpoint['quant'])


    if 'stats' in checkpoint.keys():
        stats.update(checkpoint['stats'])


    classes = torch.arange(num_classes).view(1, -1).to(device)  # the different possible classes



    def get_checkpoint():
        '''Get current checkpoint'''
        global model, stats, quant, args, optimizer, lr_scheduler, epoch

        checkpoint = {
            'model':model.state_dict(),
            'stats':stats,
            'quant': quant,
            'args' : args,
            'optimizer':optimizer.state_dict(),
            'lr_scheduler':lr_scheduler.state_dict() if lr_scheduler is not None else None,
            'epochs':epoch
                    }

        return checkpoint

    def save_checkpoint(checkpoint=None, name=None, fname=None):
        '''Save checkpoint to disk'''

        global path_output
        if name is None:
            name = "checkpoint"

        if fname is None:
            fname = os.path.join(path_output, name + '.pth')

        if checkpoint is None:
            checkpoint = get_checkpoint()

        torch.save(checkpoint, fname)


    def eval_epoch(model, dataloader):
        """Evaluate the model on the dataloader
        Return: array of [loss,error]"""


        model.eval()
        #loss_hidden_tot = np.zeros(classifier.L)  # for the
        stats = np.zeros(2)
        stats_mean = np.zeros(2)
        #loss_mean = 0
        #err_mean = 0
        #ones_hidden = torch.ones(classifier.L, device=device, dtype=dtype)

        with torch.no_grad():
            for idx, (x, y) in enumerate(dataloader):

                x = x.to(device)
                y = y.to(device)
                out_class = model(x)  # BxC,  # each output for each layer
                stats[0] = ce_loss(out_class, y).detach().cpu().numpy()  # LxTxB
                stats[1] = zero_one_loss(out_class, y).detach().cpu().numpy()  # T
                stats_mean = ((idx * stats_mean) + stats) / (idx+1)
                #err_mean = (idx * err_mean + error.detach().cpu().numpy()) / (idx+1)  # mean error
                #loss_mean = (idx * loss_mean + loss.mean(dim=-1).detach().cpu().numpy()) / (idx+1)  # mean loss
                # loss_hidden_tot = (idx * loss_hidden_tot + loss_hidden.mean(dim=1).detach().cpu().numpy()) / (idx+1)
                #break


        return stats_mean


    stop = False
    epoch = start_epoch
    separated=False
    frozen = False


    #for epoch in tqdm(range(start_epoch+1, start_epoch+args.nepoch+1)):
    # training loop
    while not stop:


        model.train()
        stats_train = np.zeros(2)
        stats = np.zeros(2)
        # 0: ce loss
        # 1: 0-1 loss aka error


        for idx, (x, y) in enumerate(train_loader):

            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device, dtype=torch.long)
            out = model(x)#, is_man)

            loss = ce_loss(out, y)

            stats[0] = loss.item()
            stats[1] = zero_one_loss(out, y).item()
            stats_train = ((idx * stats_train) + stats) / (idx+1)
            if not frozen:
                loss.backward()
                optimizer.step()


        epoch += 1 if not frozen else 0
        str_frozen= ' (frozen)' if frozen else ''



        separated = frozen and stats_train[1] == 0
        frozen = stats_train[1] == 0 and not frozen  # freeze the next epoch if 0 error

        quant.loc[epoch, ('train', 'loss')] = stats_train[0]
        quant.loc[epoch, ('train', 'error')] = stats_train[1]



        # test eval

        model.eval()
        stats_test = eval_epoch(model, test_loader)

        quant.loc[epoch, ('test', 'loss')] = stats_test[0]
        quant.loc[epoch, ('test', 'error')] = stats_test[1]

        stop = (separated
                or epoch > start_epoch + args.nepoch) # no improvement over wait epochs or total of 400 epochs



        print('ep {}, train loss (error) {:g} ({:g}), test loss (error) {:g} ({:g}) ({})'.format(
            epoch, quant.loc[epoch, ('train', 'loss')], quant.loc[epoch, ('train', 'error')],
            quant.loc[epoch, ('test', 'loss')], quant.loc[epoch, ('test', 'error')], str_frozen),
            #epoch, stats['stats_train']['ce'][-1], stats['stats_train']['zo'][-1],
            #stats['stats_test']['ce'][-1], stats['stats_test']['zo'][-1], lr_str),
            file=logs, flush=True)


        if args.lr_step>0:
            lr_scheduler.step()


        if stop or (epoch) % 5 == 0:  # we save every 5 epochs


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
                #ci=100,  # the whole spectrum of the data
                facet_kws={
                'sharey': False,
                'sharex': True
            }
            )

            g.set(yscale='log')
            plt.savefig(fname=os.path.join(path_output, 'stats.pdf'), bbox_inches="tight")
            plt.close('all')

            if args.save_model:  # we save every 5 epochs
                save_checkpoint()

    logs.close()
