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
sns.set_theme()


import models

import torch.optim
import torch
import argparse
import utils



try:
    from tqdm import tqdm
except:
    def tqdm(x): return x


if __name__ == '__main__':

    torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser('Training an annex classifier based on original VGG model')
    parser.add_argument('--name', default='annex', type=str, help='the name of the experiment')
    parser.add_argument('--learning_rate', '-lr', nargs='*', type=float, default=[5e-3], help='manual learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help="the weight decay for SGD (L2 pernalization)")
    parser.add_argument('--momentum', type=float, default=0.9, help="the momentum for SGD")
    parser.add_argument('--lr_update', '-lru', type=int, default=0, help='if any, the update of the learning rate')
    parser.add_argument('--lr_mode', '-lrm', default="num_param_tot", choices=["max", "hessian", "num_param_tot", "num_param_train", "manual"], help="the mode of learning rate attribution")
    parser.add_argument('--lr_step', '-lrs', type=int, default=30, help='the number of epochs for the lr scheduler')
    parser.add_argument('--lr_gamma',  type=float, default=0.5, help='the gamma mult factor for the lr scheduler')
    parser.add_argument('--nepoch', type=int, default=1000, help='the number of epochs to train for')
    parser.add_argument('--batch_size', '-bs', type=int, default=100, help='the dimension of the batch')
    parser.add_argument('--debug', action='store_true', help='debug')
    parser.add_argument('--ndraws', type=int, default=20, help='The number of permutations to test')
    parser.add_argument('-F', '--fraction', type=int, default=2, help='the denominator of the removed fraction of the width')
    parser_model = parser.add_mutually_exclusive_group(required=True)
    parser_model.add_argument('--model', help='path of the model to separate')
    parser_model.add_argument('--checkpoint', help='path of the previous computation checkpoint')
    parser_device = parser.add_mutually_exclusive_group()
    parser_device.add_argument('--cpu', action='store_true', dest='cpu', help='force the cpu model')
    parser_device.add_argument('--cuda', action='store_false', dest='cpu')
    parser.add_argument('--entry_layer', type=int, default=1, help='the layer ID for the tunnel entry')
    parser.set_defaults(cpu=False)



    args = parser.parse_args()

    if  len(args.learning_rate) != 2:
        args.learning_rate = 2*[args.learning_rate[0]]

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



    start_id_draw = checkpoint.get('ndraws', 0) + 1
    start_epoch = checkpoint.get('epochs', 0)

    if args.entry_layer == 0:
        args.ndraws = 1

    args_model = checkpoint_model['args']  # restore the previous arguments

    str_entry = 'entry_{}'.format(args.entry_layer)
    log_model = os.path.join(root_model, 'logs.txt')
    path_output = os.path.join(root_model, args.name, str_entry)
    # Logs


    os.makedirs(path_output, exist_ok=True)

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


    #model = utils.construct_FCN(archi)
    NUM_CLASSES = utils.get_num_classes(args_model.dataset)
    model,input_size  = models.pretrained.initialize_model(args_model.model, pretrained=False, freeze=True, num_classes=NUM_CLASSES)
    model.n_layers = utils.count_hidden_layers(model)

    transform =utils.parse_transform(log_model)

    train_dataset, test_dataset, num_chs = utils.get_dataset(dataset=args_model.dataset,
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

    try:
        new_keys = map(lambda x:x.replace('module.', ''), checkpoint_model['model'].keys())
        checkpoint_model['model'] = OrderedDict(zip(list(new_keys), checkpoint_model['model'].values()))
        #for key in state_dict.keys():
            #new_key = key.replace('module.', '')

        model.load_state_dict(checkpoint_model['model'])
        model.requires_grad_(False)
        model.eval()
    except RuntimeError as e:
        print("Can't load mode (error {})".format(e))

    #classifier = models.classifiers.Linear(model, args.ndraw, args.keep_ratio).to(device)
    #classifier = models.classifiers.ClassifierFCN(model, num_tries=args.ndraw, Rs=args.remove, =args.depth_max).to(device)
    #if args.remove is not None:
    #    remove = args.remove # the number of neurons
    #else:  # fraction is not None
    fraction = 1/args.fraction

    classifier = models.classifiers.AnnexVGG(model, F=fraction, idx_entry=args.entry_layer).to(device)




    if 'classifier' in checkpoint.keys():
        classifier.load_state_dict(checkpoint['classifier'])


    classifier.to(device)

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





    print('Annex classifier: {}'.format(str(classifier)), file=logs)
    #parameters = [ p for p in model.parameters() if not feature_extraction or p.requires_grad ]

    def zero_one_loss(x, targets):
        ''' x: TxBxC
        targets: Bx1

        returns: error of size T
        '''
        return  (x.argmax(dim=-1)!=targets).float().mean(dim=-1)

    #mse_loss = nn.MSELoss()
    #if args.ndraw == 1 or args.entry_layer == 0:  # only one try
    ce_loss = nn.CrossEntropyLoss(reduction='none')

    sets = ['train', 'test']
    stats = ['loss', 'error']
    #layers = np.arange(1, 1+1)#classifier.n_layers)  # the different layers, forward order
    ndraws = np.arange(1, start_id_draw+args.ndraws)  # the different tries

    names=['set', 'stat', 'draw']
    columns=pd.MultiIndex.from_product([sets, stats, ndraws], names=names)
    index = pd.Index(np.arange(1, start_epoch+args.nepoch+1), name='epoch')
    quant = pd.DataFrame(columns=columns, index=index, dtype=float)

    quant.sort_index(axis=1, inplace=True)  # sort for quicker access

    if 'quant' in checkpoint.keys():
        quant.update(checkpoint['quant'])



    def get_checkpoint():
        '''Get current checkpoint'''
        global model, stats, quant, args, args_model, optimizer, lr_scheduler, id_draw, epoch, stop#, params_discarded, end

        #optimizer.param_groups = optimizer.param_groups + params_discarded

        checkpoint = {
                'classifier': classifier.state_dict(),
                'stats': stats,
            'quant': quant,
                'args': args,
            'args_model': args_model,
                'optimizer': optimizer.state_dict(),
                'epochs': epoch,
            'ndraws': id_draw if stop else id_draw-1,
                    }

        #optimizer.param_groups = optimizer.param_groups[:end]

        return checkpoint


    def save_checkpoint(checkpoint=None, name=None, fname=None):
        '''Save checkpoint to disk'''

        global path_output, id_draw
        if name is None:
            name = "checkpoint_draw_{}".format(id_draw)

        if fname is None:
            fname = os.path.join(path_output, name + '.pth')


        if checkpoint is None:
            checkpoint = get_checkpoint()

        torch.save(checkpoint, fname)



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


    if not args.debug:
        logs.close()

    for id_draw in range(start_id_draw, start_id_draw+args.ndraws):

        if not args.debug:
            logs = open(os.path.join(path_output, 'logs_draw_{}.txt'.format(id_draw)), 'w')
        else:
            logs = sys.stdout


        classifier = models.classifiers.AnnexVGG(model, F=fraction, idx_entry=args.entry_layer).to(device)
        classifier.features.requires_grad_(False)
        #learning_rate = min(args.max_learning_rate, rule_of_thumb, find_learning_rate(classifier, train_loader))
        #learning_rate = rule_of_thumb
        #learning_rate = get_lr(classifier)
        #print('Learning rate: {}'.format(learning_rate), file=logs, flush=True)
        params_tot = []
        learning_rates = []
        params_feat = [p for p in classifier.features.parameters() if p.requires_grad]
        if params_feat:

            #learning_rates.append(math.sqrt(1/utils.num_parameters(classifier.features)))
            learning_rates.append(args.learning_rate[0])
            params_tot.append({'params': params_feat, 'lr': learning_rates[-1]})
        params_class = [p for p in classifier.classifier.parameters() if p.requires_grad]
        if params_class:
            #learning_rates.append(math.sqrt(1/utils.num_parameters(classifier.classifier)))
            learning_rates.append(args.learning_rate[1])
            params_tot.append({ 'params': params_class, 'lr': learning_rates[-1]})




        optimizer = torch.optim.SGD(params_tot,
                momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!", file=logs, flush=True)
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            classifier.features = nn.DataParallel(classifier.features)
            #classifier.new_sample()
            #model.classifier = nn.DataParallel(model.classifier
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)  # reduces the learning rate by half every 20 epochs

        if id_draw ==  start_id_draw:

            str_lr = "Learning rates: "
            for lr in learning_rates:
                str_lr += f'{lr:2e} '
            print(str_lr, file=logs, flush=True)
            print('Optimizer: {}'.format(optimizer), file=logs, flush=True)
            #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
            # divdes by 10 after the first epoch
            #lr_lambdas = [lambda epoch: (epoch == 1) * 1  + (epoch > 1)*1 for _ in param_list]
            #lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99, lr_min=1e-3)

            if 'optimizer' in checkpoint.keys():
                optimizer.load_state_dict(checkpoint['optimizer'])

            if 'lr_scheduler' in checkpoint.keys():
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])


        stop = False
        separated = False
        epoch = start_epoch
        frozen = False  # to freeze the next iteration

        while not stop:
        #for epoch in tqdm(range(start_epoch, start_epoch+args.nepoch)):

            classifier.train()
            #loss_hidden_tot = np.zeros(classifier.L)  # for the
            loss_train = 0 # np.zeros(args.ndraw)  # for the
            err_train = 0 #np.zeros(args.ndraw)
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
                loss.mean(dim=-1).backward()
                # loss_hidden.mean(dim=1).backward(ones_hidden)
                optimizer.step()
                        #lr_scheduler.step()



            epoch += 1 if not frozen else 0


            separated =  frozen and err_train == 0  # if already frozen and 0 error

            frozen = err_train == 0 and not frozen  # freeze the next iteration if 0 error


            # test metrics
            loss_test, err_test = eval_epoch(classifier, test_loader)


            quant.loc[pd.IndexSlice[epoch, ('train', 'error', id_draw)]] =  err_train
            quant.loc[pd.IndexSlice[epoch, ('train', 'loss', id_draw)]] =  loss_train

            quant.loc[pd.IndexSlice[epoch, ('test', 'error', id_draw)]] =  err_test
            quant.loc[pd.IndexSlice[epoch, ('test', 'loss', id_draw)]] =  loss_test


            # stopping criterion

            stop = (separated or epoch > start_epoch + args.nepoch)


            if args.lr_step >0:
                lr_scheduler.step()



            #print('ep {}, train loss (error) {:g} ({:g}), test loss (error) {:g} ({:g})'.format(
            print('try: {}, ep {}, loss (test): {:g} ({:g}), error (test): {:g} ({:g}) {}'.format(
                id_draw, epoch, quant.loc[epoch, ('train', 'loss', id_draw)], quant.loc[epoch, ('test', 'loss', id_draw)],
                err_train,  quant.loc[epoch, ('test', 'error', id_draw)], ' (separated)' if separated else ''),
                file=logs, flush=True)


            if epoch % 5 ==0 or stop:  # we save every 5 epochs

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
                    ci=None,  # the whole spectrum of the data
                    facet_kws={
                    'sharey': 'row',
                    'sharex': True
                }
                )

                g.set(yscale='log')
                #g.set(title='ds = {}, width = {}, removed = {}, Tries = {}'.format(args_model.dataset, args_model.width, args.remove, args.ndraw))
                g.fig.subplots_adjust(top=0.9, left= 0.1 )  # number of columns in the sublplot
                g.fig.suptitle('ds = {}, removed = width / {}, draw = {}, name = {}'.format(args_model.dataset, args.fraction, id_draw,  args.name))
                #g.set_axis_labels

                g.fig.tight_layout()
                plt.margins()
                plt.savefig(fname=os.path.join(path_output, 'quant_draw_{}.pdf'.format(id_draw)), bbox_inches="tight")

                plt.close('all')
                save_checkpoint()


        logs.close()

    if separated:
        sys.exit(0)  # success
    else:
        sys.exit(1)


