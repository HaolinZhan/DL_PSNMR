import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self,in_channels, out_channels, stride=1, is_shortcut=False):
        super(Block,self).__init__()
        self.is_shortcut = is_shortcut

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=21,stride=stride,padding=10,bias=False),
            nn.BatchNorm1d(out_channels ),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.4)

        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=21, stride=1, padding=10,groups=32, bias=False),
            nn.BatchNorm1d(out_channels ),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.4)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(out_channels , out_channels, kernel_size=21,stride=1,padding=10,bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.4),
        )



    def forward(self, x):
        x_shortcut = x

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x + x_shortcut

        return x



class DL_PSNMR(nn.Module):
    def __init__(self,num_classes,layer=[2,2,2]):
        super(DL_PSNMR,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.4)
        )
        self.conv2 = self._make_layer(32,32,1,num=layer[0])
        self.conv3 = self._make_layer(32,32,1,num=layer[1])
        self.conv4 = self._make_layer(32,32,1,num=layer[2])
        self.conv6=nn.Conv1d(32,1,kernel_size=1)
        self.bn6=nn.BatchNorm1d(1)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x=self.conv6(x)
        x=self.bn6(x)

        return x



    def _make_layer(self,in_channels,out_channels,stride,num):
        layers = []
        block_1=Block(in_channels, out_channels,stride=1,is_shortcut=False)
        layers.append(block_1)
        for i in range(1, num):
            layers.append(Block(out_channels,out_channels,stride=1,is_shortcut=False))
        return nn.Sequential(*layers)
