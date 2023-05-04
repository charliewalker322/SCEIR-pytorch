import torch
import torch.nn as nn
from torch.nn.functional import interpolate


class MSVector(nn.Module):

    def __init__(self, in_ch=3, std_ch=3, mean_ch=3, size=512, fea=16):
        super(MSVector, self).__init__()
        self.go = nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=fea,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4,stride=4),
            nn.Conv2d(in_channels=fea, out_channels=fea, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(in_channels=fea, out_channels=fea, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(in_channels=fea, out_channels=std_ch+mean_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.std_ch = std_ch
        self.mean_ch = mean_ch
        self.size = size

    def forward(self,x):
        x = self.resize_input(x)
        x = self.go(x)
        out_std = x[:, 0:self.std_ch]
        out_mean = x[:, self.std_ch:]
        return out_std , out_mean

    def resize_input(self,x):
        _, __, h, w = x.shape
        resize = self.size

        # # avoid upsample
        # if h*w < self.size**2:
        #     resize = self.size//2

        if h!=resize or w!=resize:
            x = interpolate(x, (resize, resize), mode='bilinear')
        return x


class Plain5(nn.Module):

    def __init__(self,in_ch,out_ch, finnal_activate=True, activate='leaky-relu', channel_attention=False, IN=False, fea=[64,64,64,64]):
        super(Plain5, self).__init__()
        if activate=='leaky-relu':
            self.activate = nn.LeakyReLU(0.2)
        else:
            self.activate = nn.ReLU()

        self.relu = nn.ReLU()
        self.finnal_activate = finnal_activate

        if IN:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=in_ch, out_channels=fea[0], kernel_size=3, stride=1, padding=1),
                self.activate,
                nn.InstanceNorm2d(num_features=fea[0],affine=True),
                nn.Conv2d(in_channels=fea[0], out_channels=fea[1], kernel_size=3, stride=1, padding=1),
                self.activate,
                nn.InstanceNorm2d(num_features=fea[1], affine=True),
                nn.Conv2d(in_channels=fea[1], out_channels=fea[2], kernel_size=3, stride=1, padding=1),
                self.activate,
                nn.InstanceNorm2d(num_features=fea[2], affine=True),
                nn.Conv2d(in_channels=fea[2], out_channels=fea[3], kernel_size=3, stride=1, padding=1),
                self.activate,
                nn.InstanceNorm2d(num_features=fea[3], affine=True),
                nn.Conv2d(in_channels=fea[3], out_channels=out_ch, kernel_size=3, stride=1, padding=1)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=in_ch, out_channels=fea[0], kernel_size=3, stride=1, padding=1),
                self.activate,
                nn.Conv2d(in_channels=fea[0], out_channels=fea[1], kernel_size=3, stride=1, padding=1),
                self.activate,
                nn.Conv2d(in_channels=fea[1], out_channels=fea[2], kernel_size=3, stride=1, padding=1),
                self.activate,
                nn.Conv2d(in_channels=fea[2], out_channels=fea[3], kernel_size=3, stride=1, padding=1),
                self.activate,
                nn.Conv2d(in_channels=fea[3], out_channels=out_ch, kernel_size=3, stride=1, padding=1)
            )

        # whether apply SE to input
        self.channel_attention = channel_attention
        if channel_attention:
            self.CA = ChannelAttentionLayer(in_ch,reduction=1)


    def forward(self,inputs):
        if self.channel_attention:
            inputs = self.CA(inputs)

        x = self.conv(inputs)
        if self.finnal_activate:
            return self.relu(x)
        else:
            return x


class ChannelAttentionLayer(nn.Module):

    def __init__(self, channel, reduction=4, feature:int=None):
        super().__init__()
        if feature:
            inner_channel = feature
        else:
            inner_channel = channel // reduction

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, inner_channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(inner_channel, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        # print(y)
        # exit('------> test done !')
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)