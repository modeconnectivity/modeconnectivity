import copy
import utils
import torch
import torch.nn as nn
from models.classifiers import RandomSamplerParallel, MultiLinear
import numpy as np
import random
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import argparse

class Path(object):

    def __init__(self, model):

        """
        Init the path from a given model (simply coy the architecutre)
        """

        self.model = copy.deepcopy(model)
        self.points = []
        self.reverse = False

    @property
    def step(self):

        return 1 if not self.reverse else -1

    def append(self, solution):
        """
        Sparsify the solution found by the subnetwork procedure
        Solution is a FCN Classifier
        """
        if type(solution) is not tuple:
            solution = (solution,)

        self.points.append(tuple(torch.nn.utils.parameters_to_vector(s.parameters()) for s in solution[::self.step]))

        pass

    def extend(self, other):

        assert (not self.reverse)
        self.points.extend(other.points[::other.step])

    def connect(self):
        """Connect the last two sparsified solutions"""
        pass

def eval_path(path, sizes, dirname):

    global device
    os.makedirs(dirname, exist_ok=True)
    param_B = path.points[0][0]
    A = copy.deepcopy(path.model).requires_grad_(False)
    AB = copy.deepcopy(path.model).requires_grad_(False)
    B = copy.deepcopy(path.model).requires_grad_(False)
    # nn.utils.vector_to_parameters(param_B, B.parameters())
    A.to(device)
    AB.to(device)
    B.to(device)
    K = 11
    lbdas = np.arange(1, K) / (K-1)
    index = pd.MultiIndex.from_product([range(len(path.points)), range(0, K-1)], names=["point", "t"])
    stats = ['loss', 'error']
    sets = ["train", "test"]
    names=['set', 'stat', 'try']
    tries = [1]
    columns=pd.MultiIndex.from_product([sets, stats, tries], names=names)
    stats = pd.DataFrame(index=index, columns=columns)
    reverse = False
    # A =  None
    # for

    for idx, pt in enumerate(path.points):
        # pt is a tuple

        if len(pt) == 1:
            param_A = pt[0].to(device)
            nn.utils.vector_to_parameters(param_A, model.parameters())  # param_A from previous iteration
            # print(f"error: {err}, loss: {loss}")
            # if idx == 0:
            loss, err = eval_epoch(model, train_loader)
            loss_test, err_test = eval_epoch(model, test_loader)
            stats.loc[(idx), 0] = loss, err, loss_test, err_test  # consider the index of the path at K-1 as the new point

            continue
        elif len(pt) == 2:
            # len(pt) == 2
            # the point is (B,C)
            # have to create intermediate point AB and walk from AB to B
            if not reverse:
                param_B = pt[0].to(device)
                # param_AB = copy.copy(A)  # from the previous iterations
                # param_A = pt[1]

                nn.utils.vector_to_parameters(param_A, A.parameters())  # param_A from previous iteration
                nn.utils.vector_to_parameters(param_B, AB.parameters())  # param_B from this iteration
                nn.utils.vector_to_parameters(param_B, B.parameters())  # param_B from this iteration
                # nn.
                AB.main[-1].weight.data = A.main[-1].weight.data
                AB.main[-1].bias.data = A.main[-1].bias.data
                param_AB = nn.utils.parameters_to_vector(AB.parameters()).to(device)
                param_A = pt[1].to(device)  # next starting point
            else:
                # reverse mode, fetch the next point
                # reverse role of AB and B, load A with next point

                param_AB = pt[1].to(device)
                param_A = path.points[idx+1][0].to(device)  # the next point
                nn.utils.vector_to_parameters(param_A, A.parameters())  # param_A from previous iteration
                nn.utils.vector_to_parameters(param_AB, AB.parameters())  # param_B from this iteration
                nn.utils.vector_to_parameters(param_AB, B.parameters())  # param_A from previous iteration

                B.main[-1].weight.data = A.main[-1].weight.data
                B.main[-1].bias.data = A.main[-1].bias.data
        # else:  # first point
                param_B = nn.utils.parameters_to_vector(B.parameters()).to(device)
            # B = pt

        elif len(pt) == 3:
            # at thispoint the last status for A is the model with
            # joint between the two paths
            reverse = True
            param_A = pt[0].to(device)
            param_B = pt[1].to(device)
            param_C = pt[2].to(device)

            nn.utils.vector_to_parameters(param_A, A.parameters())
            nn.utils.vector_to_parameters(param_B, AB.parameters())

            # nn.utils.vector_to_parameters(param_C, C.parameters())
            AB.main[-1].weight.data = A.main[-1].weight.data
            AB.main[-1].bias.data = A.main[-1].bias.data
            param_AB = nn.utils.parameters_to_vector(AB.parameters()).to(device)
            param_A = param_C

            # BC.main[-1].weight = C.main[-1].weight
            # BC.main[-1].bias = C.main[-1].bias
        for tidx, t in enumerate(lbdas, 1):
            pt = (1-t) * param_AB + t * param_B
            nn.utils.vector_to_parameters(pt, model.parameters())
            loss, err = eval_epoch(model, train_loader)
            loss_test, err_test = eval_epoch(model, test_loader)
            # print(f"error: {err}, loss: {loss}")
            stats.loc[(idx-1+tidx//(K-1), tidx%(K-1))] = loss, err, loss_test, err_test  # consider the index of the path at K-1 as the new point
            # stats.loc[(idx-1+tidx//(K-1), tidx%(K-1))] = loss_test, err_test  # consider the index of the path at K-1 as the new point

            # model.to(torch.device('cpu'))
    return stats

def plot_path(stats, quant_ds, quant_ref,  dirname):
    # df_plot = pd.melt(stats.reset_index(), id_vars=["point", "t"], ignore_index=False)
    Idx = pd.IndexSlice
    # df_plot.index.name = "index"
    for setn in ["train", "test"]:
        for stat in ["loss", "error"]:
        # df_plot = stats.loc[:, Idx[stat, :]].reset_index()
            ax = stats.plot(kind="line",
            # sns.lineplot(
                # data=df_plot,
                y=(setn,stat)
                            )
        # )
            ax.axline((0,quant_ref[stat, setn]), (1, quant_ref[stat, setn]),  ls=":", zorder=2, c='g')
            ax.axline((0,quant_ds[stat, setn]), (1, quant_ds[stat, setn]),  ls=":", zorder=2, c='r')

        plt.savefig(fname=os.path.join(dirname, f'path_{setn}_{stat}.pdf'), bbox_inches="tight")
    stats.to_csv(os.path.join(dirname, f'path.csv'))
    plt.close("all")



def read_csv(fname):
    stats = pd.read_csv(fname, header=[0,1,2], index_col=[0,1])
    stat_idx = stats.columns.names.index("stat")
    nlevels = stats.columns.nlevels
    if "err" in stats.columns.get_level_values("stat"):
        new_stat_lvl = [s.replace("err", "error") for s in stats.columns.get_level_values(stat_idx)]
        # new_stat.sort()
        levels = [stats.columns.get_level_values(i) if i != stat_idx else new_stat_lvl for i in range(nlevels)]
        cols = pd.MultiIndex.from_arrays(levels, names=stats.columns.names)
        stats.columns = cols
        stats.to_csv(fname)

    return stats






"""
Complement of a permutation with total number of elements
"""
def complement_perm(perm, total):
    idx = 0
    cperm= []
    i = 0
    while idx < total:
        while i<len(perm) and idx == perm[i]:
            idx += 1
            i+=1
        upb = perm[i] if i < len(perm) else total
        cperm.extend(list(range(idx, upb)))
        idx = upb
    return np.array(cperm)



"""
Construct the point with the new weights from the solution and with unchanged middle input layer
"""
def point_B(model, solution, ntry, idx_layer, perm, cperm):
    for idx, l in enumerate(solution.network):
        if isinstance(l, nn.ReLU):
            continue
        elif isinstance(l, nn.Linear):
            # simply copy the weights to the target model
            model.main[idx] = copy.deepcopy(l)
            # if idx == idx_layer - 2:
                # ) # also work on the bias (shift the order of the values)
        elif isinstance(l, RandomSamplerParallel):  # the random selection of features
            # the random sampling, only copy the try one
            continue
            # selection = copy.deepcopy(l.random_perms[ntry, :])
        elif isinstance(l, MultiLinear):  # after the selection of features
            # weights and bias of the original model
            # different case if at the layer of starting point
            weight = (model.main[idx-1].weight)  # the previous weight, should be block diagonal
            bias = (model.main[idx-1].bias)
            wl = l.weight[ntry, :, :].transpose(0, 1)  # the (transposed) weight of the solution
            sz_wl = wl.size()
            sz_w = weight.size()
            m = sz_wl[0]  # number of the dropped units
            p = sz_wl[1]
            # n = sz_w[0]
            # d = sz_w[1]
            # if idx == idx_layer+1:  # modification of the previous layer
                # at the selection layer
                #construct the new weight with first rows fully connected and then solution | 0
                # pass
            if idx == idx_layer+1:  # the actual weight to modify

                # will copy wl to the bottom rows of weight
                weight[-m:, perm] = wl  # no block 0? only fill in the weights in the permutation
                weight[-m:, cperm] = 0  # block 0, only on the correct weights
                # copy the bias values at the correct location, keep the
                # original on top
                bias[-m:] = l.bias[ntry, 0, :]
                # weight[:, :-p].zero_()   # output weight to 0
            else:  # for indices after the first layer
                weight[-m:,-p:] = wl
                bias[-m:] = l.bias[ntry, 0, :]

                if idx == len(solution.network) - 1:
                    weight[:, :-p].zero_()  # set the output weight to zero


                # bias[:, d-p].zero_()   # output weight to 0
            # model.main[idx-1].weight = nn.Parameter(weight)
            # model.main[idx-1].biais = nn.Parameter(bias)
        else:
            pass
    return model

"""
construct the model modifying the input layer so that it can be the starting point for the next model
"""

def point_C(model, solution, ntry, idx_layer, perm, cperm):
    for idx, l in enumerate(solution.network):
        if isinstance(l, nn.ReLU):
            continue
        elif isinstance(l, nn.Linear):

            if idx == idx_layer - 2:
                weight = model.main[idx].weight
                bias = model.main[idx].bias
                weight[:len(perm), : ] = weight[perm]
                weight[len(perm):, :] = 0
                bias[:len(perm) ] = bias[perm]
                bias[len(perm):] = 0
            # model.main[idx] = copy.deepcopy(l)
                # model.main[idx].weight = nn.Parameter(weight)
                # model.main[idx].bias = nn.Parameter(bias)
            else:
                continue
            # if idx == idx_layer - 2:
                         # l.bias[cperm]],
                    # dim=0)
                # ) # also work on the bias (shift the order of the values)
        elif isinstance(l, RandomSamplerParallel):  # the random selection of features
            # the random sampling, only copy the try one
            continue
            # selection = copy.deepcopy(l.random_perms[ntry, :])
        elif isinstance(l, MultiLinear):  # after the selection of features
            # weights and bias of the original model
            # different case if at the layer of starting point
            weight = (model.main[idx-1].weight)  # the previous weight, should be block diagonal
            bias = (model.main[idx-1].bias)
            wl = l.weight[ntry, :, :].transpose(0, 1)  # the (transposed) weight of the solution
            sz_wl = wl.size()
            sz_w = weight.size()
            m = sz_wl[0]
            p = sz_wl[1]
            n = sz_w[0]
            d = sz_w[1]  # total dimension
            # p = sz_w[1] - sz_wl[0] # size of the permutation
            if idx == idx_layer+1:  # modification of the previous layer
                # have to reorder the output units so that the removed ones are
                # at the bottom
                # put the xi parameters to the top
                # wl = torch.cat([wl, torch.zeros(m, d - p)], dim=1)
                weight.zero_()
                weight[:m, :p] = wl
                bias.zero_()
                bias[:m] = l.bias[ntry, 0, :]
                # at the selection layer
                #construct the new weight with first rows fully connected and then solution | 0
                # pass
            # elif idx == idx_layer+3:  # the actual weight to modify
                # have to permute the weights
                # weight[:m, :len(perm)] = xi[perm, :]  #

                # weight = (model.main[idx-1].weight)  # the previous weight, should be block diagonal
                # bias = (model.main[idx-1].bias)
                # will copy wl to the bottom rows of weight
                # bias[m:, :] = l.bias[ntry, 1, :]
            else:  # for indices after the first layer
                # xi = copy.copy(weight[-m:, -p:])
                # save previous value
                xi = weight[-m:, -p:].clone()
                weight.zero_()
                # put it on "top"
                weight[:m, :p] = xi

                # save previous value
                b = bias[-m:].clone()
                bias.zero_()
                # put it on "top"
                bias[:m] = b

    return model

def sparsify(solution, model, path, ntry=1):
    # requires a model and  to copy the different weights into it ?
    # record two new points in the path, cf scheme number 3 and 6
    # assume the model being in the correct previous configuration, i.e. the
    # I_p neurons are non zero and the others are zero for p \in {l+1, ...,
    # L-1}
    selection = None
    idx_layer = [isinstance(l, RandomSamplerParallel) for l in solution.network].index(True)
    perm = solution.network[idx_layer].random_perms[ntry, :].view(-1).numpy()
    total = solution.network[idx_layer].N
    cperm = complement_perm(perm, total)

    B = copy.deepcopy(point_B(model, solution, ntry, idx_layer, perm, cperm))
    B.requires_grad_(False)
    # path.append(model)
    C = copy.deepcopy(point_C(model, solution, ntry, idx_layer, perm, cperm))
    C.requires_grad_(False)
    path.append((B,C))


    return path

"""
Connects the last points for a path
"""
def last_layer(solution, model, path):
    sizes = []
    for idx, l in enumerate(solution.network):
        if isinstance(l, nn.Linear):
            m, p = l.weight.size()
            sizes.append(p)
            model.main[idx].weight[-m:, -p:] = l.weight
            model.main[idx].bias[-m:] = l.bias
            if idx == len(solution.network) - 1:
                sizes.append(m)
                model.main[idx].weight[:m, :p].zero_()
                model.main[idx].bias[:m].zero_()

    B = copy.deepcopy(model)
    C = copy.deepcopy(downsideup(model, sizes))
    path.append((B, C))
    return path



"""
m1 and m2 are two models (sparsified)
"""
def connect_two_models(path, model, target, sizes):


    global device
    A = copy.deepcopy(model)
    # upsidedown(target, sizes)
    B = copy.deepcopy(flip_copy_incoming(model, target, sizes))
    # path.append(model)
    C = copy.deepcopy(downsideup(model, sizes))
    path.append((A,B,C))
    # model.to(devj,c ce)
    # loss, err = eval_epoch(model, train_loader)
    # print(f"error: {err}, loss: {loss}")
    # model.to(torch.device('cpu'))
    return path


"""
swap top and bottom neurons in the model, set the bottom to 0
"""
def downsideup(model, sz):
    idx = 1
    for l in model.main:
        if isinstance(l, nn.Linear):
            nin, nout = sz[idx-1], sz[idx]
            l.weight[:nout, :nin] = l.weight[-nout:, -nin:]
            l.weight[-nout:, -nin:].zero_()
            l.bias[:nout]  = l.bias[-nout:]
            l.bias[-nout:].zero_()
            idx += 1
        else:
            continue

    return model


"""
copy incoming connections for the layers from target to model
assume the weights of target are on the up side and do not intersect with the weights of the model
"""
def flip_copy_incoming(model, target, sz):
    sidx = 1
    for lidx, layer in enumerate(model.main):
        if isinstance(layer, nn.Linear):
            nin, nout = sz[sidx-1], sz[sidx]
            layer.weight[-nout:, -nin:] = target.main[lidx].weight[:nout, :nin]
            layer.bias[-nout:]  = target.main[lidx].bias[:nout]
            # layer.bias[:nout].zero_()
            sidx += 1
            if lidx == len(model.main) -1 :
                layer.weight[:nout, :nin].zero_()
        else:
            continue

    return model


    #

ce_loss = nn.CrossEntropyLoss(reduction='none')

def zero_one_loss(x, targets):
    ''' x: TxBxC
    targets: Bx1

    returns: err of size T
    '''
    return  (x.argmax(dim=-1)!=targets).float().mean(dim=-1)

def select_try(model, solution, ntry):

    idx_layer = [isinstance(l, RandomSamplerParallel) for l in solution.network].index(True)
    perm = solution.network[idx_layer].random_perms[ntry, :].view(-1).numpy()
    total = solution.network[idx_layer].N
    cperm = complement_perm(perm, total)
    for idx, l in enumerate(solution.network):
        if isinstance(l, RandomSamplerParallel):
            pass
        elif isinstance(l, MultiLinear):
            m, p = l.weight.size()
            model.main[idx-1].weight.zero_()
            model.main[idx-1].weight.data[:m, :p] = l.weight.data
            pass
    return model



def eval_epoch(model, dataloader, ntry=None):


    global device
    model.eval()
    model.to(device)
    #loss_hidden_tot = np.zeros(classifier.L)  # for the
    loss_mean = 0
    err_mean = 0
    #ones_hidden = torch.ones(classifier.L, device=device, dtype=dtype)

    with torch.no_grad():
        for idx, (x, y) in enumerate(dataloader):

            x = x.to(device)
            y = y.to(device)
            out_class = model(x)  # BxC,  # each output for each layer
            if ntry is not None and out_class.dim() == 3:
                out_class = out_class[ntry, :, :]
            loss = ce_loss(out_class, y)  # LxTxB
            err = zero_one_loss(out_class, y)  # T
            err_mean = (idx * err_mean + err.detach().cpu().numpy()) / (idx+1)  # mean error
            loss_mean = (idx * loss_mean + loss.mean(dim=-1).detach().cpu().numpy()) / (idx+1)  # mean loss
            # loss_hidden_tot = (idx * loss_hidden_tot + loss_hidden.mean(dim=1).detach().cpu().numpy()) / (idx+1)
            #break


    model.to(torch.device('cpu'))
    return loss_mean, err_mean

if __name__ == "__main__":

    parser = argparse.ArgumentParser('Creation and evaluation of a path connecting two solutions')
    parser_device = parser.add_mutually_exclusive_group()
    parser_device.add_argument('--cpu', action='store_true', dest='cpu', help='force the cpu model')
    parser_device.add_argument('--cuda', action='store_false', dest='cpu')
    parser.add_argument('--nameA', default='A', help = "name of the experiment A folder")
    parser.add_argument('--nameB', default='B', help = "name of the experiment B folder")
    parser.add_argument('--M1', help="the first model to connect (checkpoint)")
    parser.add_argument('--M2', help="the second model to connect (checkpoint)")
    parser.add_argument('--output', help="directory for outputs (if None will be where the original models were)")
    parser.set_defaults(cpu=False)

    args = parser.parse_args()
    use_cuda = torch.cuda.is_available() and not args.cpu
    num_gpus = torch.cuda.device_count()  # random index for the GPU
    gpu_index = random.choice(range(num_gpus)) if num_gpus > 0  else 0
    device = torch.device('cuda' if use_cuda else 'cpu', gpu_index)

    fn_log_model = os.path.join(os.path.dirname(args.M1), 'logs.txt')
    archi_model = utils.parse_archi(fn_log_model)
    fn_model =  args.M1
    chkpt_model = torch.load(fn_model, map_location=lambda storage, location: storage)
    model = copy.deepcopy(utils.construct_FCN(archi_model))
    path = Path(model)
    args_model = chkpt_model["args"]
    # args_model = chkpt_model["args"]
    # model.requires_grad_(False)
    # selsol =model
    # path.extend(selsol)
    n_layer = utils.count_hidden_layers(model)
    ntry = 1

    imresize=None
    train_dataset, test_dataset, num_chs = utils.get_dataset(dataset=args_model.dataset,
                                                          dataroot=args_model.dataroot,
                                                             imresize =imresize,
                                                            normalize= args_model.normalize if hasattr(args_model, 'normalize') else False,
                                                             )
    # print('Transform: {}'.format(train_dataset.transform), file=logs, flush=True)
    train_loader, size_train,\
        test_loader, size_test  = utils.get_dataloader( train_dataset,
                                                       test_dataset, batch_size
                                                       =args_model.batch_size,
                                                       size_max=100, #args_model.size_max,
                                                       collate_fn=None,
                                                       pin_memory=True)
    paths = dict()
    models = dict()
    quant_ref = pd.DataFrame()
    quant_ds = pd.DataFrame()
    Idx = pd.IndexSlice
    for mid, fn_model in enumerate([args.M1, args.M2], 1):

        models[mid] = copy.deepcopy(utils.construct_FCN(archi_model))  # copy the model
        model = models[mid]
        dir_model = os.path.dirname(fn_model)
        dir_expA = os.path.join(dir_model, args.nameA)  # the directories of the experiments
        dir_expB = os.path.join(dir_model, args.nameB)

        fn_ds =  os.path.join(dir_expB, "eval_copy.pth")  # the filename for the experiment B
        chkpt_ds = torch.load(fn_ds, map_location=lambda storage, loc: storage)
        quant = chkpt_ds["quant"]
        quant_ref = pd.concat([quant_ref, quant.loc[1, Idx[0, :, :]].to_frame().transpose()], ignore_index=True, axis=0)
        idx_max = quant.loc[:, Idx[1:, "loss", "train"]].idxmax(axis=1)
        idx_ds = quant[idx_max].idxmin()
        quant_ds = pd.concat([quant_ds, quant.loc[idx_ds[1][0], Idx[idx_ds.keys()[1][0], :, :]].to_frame().transpose()], ignore_index=True, axis=0)  # select the step and the layer that define the bound
        chkpt_model = torch.load(fn_model, map_location=lambda storage, loc: storage)
        model.load_state_dict(chkpt_model['model'])
        paths[mid] = Path(model)
        path = paths[mid]
        path.reverse = mid == 2
        path.append(model)  # first points
        # args_model = chkpt_model["args"]
        model.requires_grad_(False)
        for eid in range(n_layer, -1, -1):
            fn_log_sol =  os.path.join(dir_expA, f"logs_entry_{eid}.txt")
            fn_solution = os.path.join(dir_expA, f"checkpoint_entry_{eid}.pth")
            chkpt = torch.load(fn_solution, map_location=lambda storage, loc: storage)
            archi_sol = utils.parse_archi(fn_log_sol)
            solution = utils.construct_classifier(archi_sol)
            solution.load_state_dict(chkpt['classifier'])
            solution.requires_grad_(False)
            # eval_epoch(solution, train_loadr, ntry)
            if eid == 0:
                path = last_layer(solution, model, path)
            else:
                path = sparsify(solution, model, path, ntry)
            # model.to(device)
            # err, loss = eval_epoch(model, train_loader)
            # print(f"error: {err}, loss: {loss}")
            # model.to(torch.device('cpu'))
    #both paths are computed

    quant_ds.index = [1,2]
    quant_ref.index = [1,2]
    quant_ds = quant_ds.mean().droplevel("layer")
    quant_ref = quant_ref.mean().droplevel("layer")
    sizes = [l.in_features for l in solution.network if isinstance(l, nn.Linear)] + [solution.network[-1].out_features]
    connect_two_models(paths[1], models[1], models[2], sizes)

    paths[1].extend(paths[2])  # terminate the path 1 with path 2
    dname = args.output if args.output is not None else os.path.join(os.path.commonpath([args.M1,args.M2]), "path")
    os.makedirs(dname, exist_ok=True)
    quant_ds.to_csv(os.path.join(dname, "B.csv"))
    quant_ref.to_csv(os.path.join(dname, "ref.csv"))
    stats = eval_path(paths[1], sizes, dirname=dname)
    stats.to_csv(os.path.join(dname, f'A.csv'))






