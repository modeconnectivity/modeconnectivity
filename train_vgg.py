
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
    parser.add_argument('--dataroot', '-droot', default='./data/', help='the root for the input data')
    parser.add_argument('--num_wokers', '-j', type=int, default=4, help='num of processor workers for loading the data')
    parser.add_argument('--name', default='vgg', type=str, help='the name of the experiment')
    parser.add_argument('--learning_rate', '-lr', type=float, nargs='*', default=[5e-3], help='leraning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help="the weight decay for SGD (L2 pernalization)")
    parser.add_argument('--momentum', type=float, default=0.9, help="the momentum for SGD")
    parser.add_argument('--lr_step', '-lrs', type=int, default=0, help='the step for the learning rate scheduler')
    parser.add_argument('--lr_gamma', '-lrg', type=float, default=0.5, help='the step for the learning rate scheduler')
    parser.add_argument('--save_model', action='store_true', default=True, help='stores the model after some epochs')
    parser.add_argument('--nepoch', type=int, default=1000, help='the number of epochs to train for')
    parser.add_argument('--batch_size', '-bs', type=int, default=100, help='the dimension of the batch')
    parser.add_argument('--debug', action='store_true', help='debug')
    parser.add_argument('--output_root', '-oroot', help='the root path for the outputs')
    parser.add_argument('--vary_name', nargs='*', default=None, help='the name of the parameter to vary in the name (appended)')
    #parser.add_argument('--keep_ratio', type=float, default=0.5, help='The ratio of neurons to keep')
    parser_model = parser.add_mutually_exclusive_group(required=True)
    parser_model.add_argument('--model', choices=['vgg-16', 'vgg-11'], help='the type of the model to train')
    parser_model.add_argument('--checkpoint', help='path of the previous computation checkpoint')
    parser_device = parser.add_mutually_exclusive_group()
    parser_device.add_argument('--cpu', action='store_true', dest='cpu', help='force the cpu model')
    parser_device.add_argument('--cuda', action='store_false', dest='cpu')
    parser.set_defaults(cpu=False)
    parser_feat = parser.add_mutually_exclusive_group()
    parser_feat.add_argument('--feature_extract', action='store_true', dest='feature_extract', help='use the pretrained model as a feature extractor')
    parser_feat.add_argument('--no-feature_extract', action='store_false', dest='feature_extract')
    parser.set_defaults(feature_extract=False)



    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    #device = torch.device('cpu')

    if  len(args.learning_rate) != 2:
        args.learning_rate = 2*[args.learning_rate[0]]

    if args.output_root is None:
        args.output_root = utils.get_output_root(args)
        # default output directory

    if args.vary_name is not None:
        args.name = utils.get_name(args)

    dtype = torch.float
    num_gpus = torch.cuda.device_count()

    if args.checkpoint is not None:
        try:
            checkpoint = torch.load(args.checkpoint, map_location=device)
            #args = checkpoint['args']
            args.__dict__.update(checkpoint['args'].__dict__)

            #cont = True  # proceed the computation
        except RuntimeError:
            print('Could not load the model')


    else:
        checkpoint = dict()



    path_output = os.path.join(args.output_root, args.name)

    os.makedirs(path_output, exist_ok=True)


    if not args.debug:
        logs = open(os.path.join(path_output, 'logs.txt'), 'w')
    else:
        logs = sys.stdout

    # Logs
    NUM_CLASSES = utils.get_num_classes(args.dataset)
    log_fname = os.path.join(args.output_root, args.name, 'logs.txt')

    feature_extract=args.feature_extract
    model, input_size = models.pretrained.initialize_model(args.model, pretrained=False, feature_extract=feature_extract, num_classes=NUM_CLASSES)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!", file=logs, flush=True)
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model.features = nn.DataParallel(model.features)
        #model.classifier = nn.DataParallel(model.classifier)


    #model.to(device)
    model.cuda()

    if 'model' in checkpoint.keys():
        model.load_state_dict(checkpoint['model'])

    m_feats = model.features #if isinstance(model, nn.DataParallel) else model.features
    m_class = model.classifier #if isinstance(model, nn.DataParallel) else model.classifier



    #logs_debug = open(os.path.join(path_output, 'debug.log'), 'w')
#     logs = None

    print(os.sep.join((os.path.abspath(__file__).split(os.sep)[-2:])), file=logs)  # folder + name of the script
    print('device= {}, num of gpus= {}'.format(device, num_gpus), file=logs)
    print('dtype= {}'.format(dtype), file=logs)

    for k, v in vars(args).items():
        print("%s= %s" % (k, v), file=logs, flush=True)

    #imresize = (256, 256)
    #imresize=(64,64)
    train_dataset, test_dataset, num_chs = utils.get_dataset(dataset=args.dataset,
                                                          dataroot=args.dataroot,
                                                                            normalize=True,
                                                             )
    train_loader, size_train,\
        test_loader, size_test  = utils.get_dataloader( train_dataset, test_dataset,
                                                       batch_size =args.batch_size, num_workers=4,
                                                       collate_fn=None, pin_memory=True)

    #model = models.cnn.CNN(1)

    num_classes = len(train_dataset.classes) if args.dataset != 'svhn' else 10
    imsize = next(iter(train_loader))[0].size()[1:]




    num_parameters = utils.num_parameters(model)
    num_samples_train = size_train
    num_samples_test = size_test
    print('Number of parameters: {}'.format(num_parameters), file=logs)
    print('Number of training samples: {}'.format(num_samples_train), file=logs)
    print('Number of testing samples: {}'.format(size_test), file=logs)
    #print('Layer dimensions'.format(linear_classifier.neurons), file=logs)
    print('Image dimension: {}'.format(imsize), file=logs)

    #model.apply(models.cnn.init_weights)



    ce_loss = nn.CrossEntropyLoss()



    if not feature_extract:
        optimizer = torch.optim.SGD([
            {'params': m_feats.parameters(), 'lr': args.learning_rate[0]},
                {'params': m_class.parameters(), 'lr': args.learning_rate[1]}],
                momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD([
                {'params': m_class.parameters()}],
                #{'params': model.features.parameters(), 'lr': 1e-5}],
                lr=args.learning_rate[0], momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)

    #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)  # reduces the learning rate by half every 20 epochs
    #lr_scheduler = None

    print("Model: {}".format(model), file=logs, flush=True)
    print("Optimizer: {}".format(optimizer), file=logs, flush=True)
    print("LR Scheduler: {}".format(lr_scheduler), file=logs, flush=True)

    if 'optimizer' in checkpoint.keys():
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except ValueError as e:
            print(e)


    if 'lr_scheduler' in checkpoint.keys():
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    start_epoch = checkpoint.get('epochs', 0)

    names=['set', 'stat']
    #tries = np.arange(args.ndraw)
    sets = ['train', 'test']
    stats = ['loss', 'error']
    #layers = ['last', 'hidden']
    columns=pd.MultiIndex.from_product([sets, stats], names=names)
    index = pd.Index(np.arange(1, args.nepoch+start_epoch), name='epoch')
    quant = pd.DataFrame(columns=columns, index=index, dtype=float)


    if 'quant' in checkpoint.keys():
        checkpoint_saved = checkpoint['quant']
        quant.loc[1:start_epoch+1, :] = checkpoint_saved.loc[1:start_epoch+1, :]


    classes = torch.arange(num_classes).view(1, -1).to(device)  # the different possible classes

    def zero_one_loss(x, targets):
        ''' x: BxC
        targets: Bx1

        returns: error of dim 0
        '''
        return  (x.argmax(dim=1)!=targets).float().mean(dim=0)

    #mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()



    def get_checkpoint():

        global epoch
        global model
        global args
        global optimizer

        checkpoint = {'model':model.state_dict(),
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


    stop = False
    epoch = start_epoch
    separated=False
    frozen = False  # will freeze the update to check if data is separated

    def eval_epoch(model, dataloader):


        model.eval()
        #loss_hidden_tot = np.zeros(classifier.L)  # for the
        loss_mean = 0
        err_mean = 0
        #ones_hidden = torch.ones(classifier.L, device=device, dtype=dtype)

        with torch.no_grad():
            for idx, (x, y) in enumerate(dataloader):

                x = x.to(device)
                y = y.to(device)
                out_class = model(x)  # BxC,  # each output for each layer
                loss = ce_loss(out_class, y)  # LxTxB
                error = zero_one_loss(out_class, y)  # T
                err_mean = (idx * err_mean + error.detach().cpu().numpy()) / (idx+1)  # mean error
                loss_mean = (idx * loss_mean + loss.mean(dim=-1).detach().cpu().numpy()) / (idx+1)  # mean loss
                # loss_hidden_tot = (idx * loss_hidden_tot + loss_hidden.mean(dim=1).detach().cpu().numpy()) / (idx+1)
                #break


        return loss_mean, err_mean



    while not stop:


        loss_train =  0
        err_train = 0
        #loss_hidden_tot = np.zeros(args.ndraw)  # for the
        #ones = torch.ones(args.ndraw, device=device, dtype=dtype)

        for idx, (x, y) in enumerate(train_loader):


            x = x.to(device)
            y = y.to(device)

            for p in model.parameters():
                p.grad=None

            out = model(x)  #  BxC  # each output for each layer
            loss = ce_loss(out, y).mean()  # TxB

            loss_train = (idx * loss_train + loss.detach().cpu().numpy()) / (idx+1)
            error = zero_one_loss(out,y)
            err_train = (idx * err_train + error.detach().cpu().numpy() )/ (idx+1)
            if not frozen:
                loss.backward()
                optimizer.step()

        loss_test, err_test = eval_epoch(model, test_loader)

        if epoch == start_epoch:  # first epoch
            loss_min = loss_test
            err_min = err_test

        epoch += 1 if not frozen else 0


        quant.loc[epoch, ('train', 'error')] = err_train
        quant.loc[epoch, ('train', 'loss')] = loss_train

        quant.loc[epoch, ('test', 'error')] = err_test
        quant.loc[epoch, ('test', 'loss')] = loss_test

        separated =  frozen and err_train == 0
        frozen = err_train == 0  and not frozen # will test with frozen network next time, prevent from freezing twice in a row

        if frozen:
            print("Freezing the next iteration", file=logs, flush=True)



        stop = ( separated
                or epoch > start_epoch + args.nepoch)



        print('ep {}, train loss (error) {:g} ({:g}), test loss (error) {:g} ({:g})'.format(
            epoch, quant.loc[epoch, ('train', 'loss')], quant.loc[epoch, ('train', 'error')],
            quant.loc[epoch, ('test', 'loss')], quant.loc[epoch, ('test', 'error')]),
            file=logs, flush=True)

        if args.lr_step>0:
            lr_scheduler.step()


        if epoch%5 == 0 or stop:
            quant_reset = quant.reset_index()
            quant_plot = pd.melt(quant_reset, id_vars='epoch')
            g = sns.relplot(
                data = quant_plot,
                #col='layer',
                hue='set',
                col='stat',
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

            plt.savefig(fname=os.path.join(path_output, 'losses.pdf'))


            plt.close('all')

            if args.save_model:  # we save every 5 epochs
                save_checkpoint()


    logs.close()
    #logs_debug.close()

    #save_checkpoint()
    sys.exit(0)  # success

