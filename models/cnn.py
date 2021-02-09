import torch.nn as nn
import torch


class CNN3(nn.Module):

    def __init__(self, input_depth=1, depth=64, lrelu=0.01):

        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(input_depth, depth, kernel_size=3, stride=1),
            nn.LeakyReLU(lrelu, inplace=True),
            #nn.BatchNorm2d(64),
            nn.Conv2d(depth, depth, kernel_size=3, stride=1),
            nn.LeakyReLU(lrelu, inplace=True),
            nn.Conv2d(depth, depth, kernel_size=3, stride=1),
            nn.LeakyReLU(lrelu, inplace=True),
            nn.Conv2d(depth, depth*2, kernel_size=3, stride=2),
            #nn.MaxPoold2d(kernel_size=2),
            nn.LeakyReLU(lrelu, inplace=True),
            nn.Conv2d(depth*2, depth*2, kernel_size=3, stride=1),
            nn.LeakyReLU(lrelu, inplace=True),
            nn.Conv2d(depth*2, depth*2, kernel_size=3, stride=1),
            nn.LeakyReLU(lrelu, inplace=True),
            nn.Conv2d(depth*2, depth*4, kernel_size=3, stride=2),
            nn.LeakyReLU(lrelu, inplace=True),
            nn.Conv2d(depth*4, depth*4, kernel_size=3, stride=1),
            nn.LeakyReLU(lrelu, inplace=True),
            nn.Conv2d(depth*4, depth*8, kernel_size=3, stride=2),
            nn.LeakyReLU(lrelu, inplace=True),
            nn.Conv2d(depth*8, depth*8, kernel_size=3, stride=1),
            nn.LeakyReLU(lrelu, inplace=True),
            nn.Conv2d(depth*8, depth*8, kernel_size=3, stride=2),
            nn.LeakyReLU(lrelu, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),

        )

        self.fc = nn.Sequential(
            nn.Linear(depth*8+ 1, depth*16),
            nn.LeakyReLU(lrelu, inplace=True),
            nn.Linear(depth*16, depth*16),
            nn.LeakyReLU(lrelu, inplace=True),
            nn.Linear(depth*16, 1)

            #nn.LeakyReLU(0.01, True),
        )


    def forward(self, x, man):

        feats = torch.cat((man.float().view(-1, 1), self.main(x).view(x.size(0), -1)), dim=1)
        out = self.fc(feats)
        return out


    def num_parameters(self, only_require_grad=False):
        '''Return the number of parameters in the model'''
        return sum(p.numel() for p in self.parameters() if not only_require_grad or p.requires_grad)

class CNN(nn.Module):

    def __init__(self, input_depth=1, depth=64):

        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(input_depth, 64, kernel_size=21, stride=10),
            #nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=11, stride=5),
            #nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=5, stride=2),
            #nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size =5, stride=2),
            #nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, kernel_size =3, stride=2),
            #nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Conv2d(512, 1024, kernel_size = 3, stride=1),
            nn.Conv2d(512, 1024, kernel_size =2, stride=2),
            #nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.LeakyReLU(0.01, inplace=True)
            nn.AdaptiveAvgPool2d((1, 1)),

        )

        self.fc = nn.Sequential(
            nn.Linear(1024+ 1, 1),
            #nn.LeakyReLU(0.01, True),
        )


    def forward(self, x, man):

        feats = torch.cat((man.float().view(-1, 1), self.main(x).view(x.size(0), -1)), dim=1)
        out = self.fc(feats)
        return out


    def num_parameters(self, only_require_grad=False):
        '''Return the number of parameters in the model'''
        return sum(p.numel() for p in self.parameters() if not only_require_grad or p.requires_grad)



class SmallCNN(CNN):


    def __init__(self, input_depth=1, depth=64):

        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(input_depth, depth, kernel_size=21, stride=10),
            #nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            #nn.Conv2d(64, 128, kernel_size=11, stride=5),
            #nn.BatchNorm2d(128),
            #nn.LeakyReLU(0.2, inplace=True),
            #nn.Conv2d(128, 256, kernel_size=5, stride=2),
            #nn.BatchNorm2d(256),
            #nn.LeakyReLU(0.2, inplace=True),
            #nn.Conv2d(256, 512, kernel_size =5, stride=2),
            #nn.BatchNorm2d(512),
            #nn.LeakyReLU(0.2, inplace=True),
            #nn.Conv2d(512, 512, kernel_size =3, stride=2),
            #nn.BatchNorm2d(512),
            #nn.LeakyReLU(0.2, inplace=True),
            #nn.Conv2d(512, 1024, kernel_size = 3, stride=1),
            #nn.Conv2d(512, 1024, kernel_size =2, stride=2),
            #nn.BatchNorm2d(512),
            #nn.LeakyReLU(0.2, inplace=True),
            #nn.LeakyReLU(0.01, inplace=True)
            nn.AdaptiveAvgPool2d((1, 1)),

        )

        self.fc = nn.Sequential(
            nn.Linear(depth + 1, depth // 2),
            nn.LeakyReLU(0.2, True),
            nn.Linear(depth//2 , 1),
            #nn.LeakyReLU(0.01, True),
        )


    def forward(self, x, man):

        feats = torch.cat((man.float().view(-1, 1), self.main(x).view(x.size(0), -1)), dim=1)
        out = self.fc(feats)
        return out


    def num_parameters(self, only_require_grad=False):
        '''Return the number of parameters in the model'''
        return sum(p.numel() for p in self.parameters() if not only_require_grad or p.requires_grad)

class MaxPoolCNN(CNN):


    def __init__(self, input_depth=1, depth=64):
        '''input_size (int, int): HxW of the (resized) input
        '''

        super().__init__()



        self.main = nn.Sequential(
            nn.Conv2d(input_depth, depth, kernel_size=7, stride=1),
            #nn.BatchNorm2d(128),
            #nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=8),
            nn.Conv2d(depth, depth*2, kernel_size=5, stride=1),
            #nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=4),

            #nn.Conv2d(depth*2, depth*4, kernel_size =3, stride=2),
            #nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(depth*2, depth*4, kernel_size=6, stride=1),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Conv2d(depth*2, depth*4, kernel_size =3, stride=2),
            #nn.BatchNorm2d(512),
            #nn.LeakyReLU(0.2, inplace=True),
            #nn.Conv2d(512, 1024, kernel_size = 3, stride=1),
            #nn.LeakyReLU(0.01, inplace=True)

        )

        self.fc = nn.Sequential(
            nn.Linear(depth*4+1, 10),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(10, 1)
        )


        return

class ResizeCNN(CNN):

    def __init__(self, input_size, input_depth=1, depth=64):
        '''input_size (int, int): HxW of the (resized) input
        '''

        super().__init__()


        h, w = input_size

        self.main = nn.Sequential(
            nn.Conv2d(input_depth, depth, kernel_size=11, stride=5),
            #nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(depth, depth*2, kernel_size=5, stride=2),
            #nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(depth*2, depth*4, kernel_size =3, stride=2),
            #nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Conv2d(512, 1024, kernel_size = 3, stride=1),
            #nn.LeakyReLU(0.01, inplace=True)

        )

        h_out = (((((h - 11) // 5 + 1) - 5) // 2 + 1) - 3 )//2 + 1
        w_out = (((((w - 11) // 5 + 1) - 5) // 2 + 1) - 3 )//2 + 1

        # output of size depth*4 x h x w

        self.fc = nn.Sequential(
            nn.Linear(h_out * w_out * depth*4+ 1, 1),
            #nn.LeakyReLU(0.01, True),
        )


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        torch.nn.init.xavier_uniform_(m.weight)
        if hasattr(m, 'bias'):
            m.bias.data.zero_()


