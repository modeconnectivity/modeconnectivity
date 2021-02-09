import torch.nn as nn
import torch
import utils

class FCN(nn.Module):

    def __init__(self, input_dim, depth=1024, lrelu=0.01):

        super().__init__()

        sizes = [input_dim, depth, 10]
        self.main = utils.construct_mlp_net(sizes, fct_act=nn.LeakyReLU, kwargs_act={'negative_slope': 0.2, 'inplace': True})

        self.main.append(nn.Softmax)

        return




    def forward(self, x):

        #vec = torch.cat((is_man.view(-1, 1), x.view(x.size(0), -1)), dim=1)

        out = self.main(x.view(x.size(0), -1))
        return out


    def num_parameters(self, only_require_grad=False):
        '''Return the number of parameters in the model'''
        return sum(p.numel() for p in self.parameters() if not only_require_grad or p.requires_grad)



