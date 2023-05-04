import torch
import torch.nn as nn
from torchvision import models


class perceptual_loss(nn.Module):
    def __init__(self, device, weights=None):
        super().__init__()
        self.vgg19 = VGG19().to(device)
        self.l1 = nn.L1Loss()
        if not weights:
            self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]  # 可调整权重
        else:
            self.weights = weights

    
    def forward(self, i, i_hat):
        i_fs = self.vgg19(i)
        i_hat_fs = self.vgg19(i_hat)
        loss = 0
        for i in range(0, len(i_fs)):
            loss += self.weights[i] * self.l1(i_fs[i], i_hat_fs[i])
        return loss


class VGG19(nn.Module):
    def __init__(self):
        super().__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()

        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.required_grad = False

    def forward(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class Charbonnier_loss(torch.nn.Module):

    def __init__(self,eps=1e-3):
        super(Charbonnier_loss, self).__init__()
        self.eps = eps

    def forward(self, x, y,sepa_channel=False):
        if sepa_channel:
            loss = 0
            for i in range(x.shape[1]):
                loss += torch.sqrt( (x-y)**2 + self.eps ).mean()
            return loss.mean()
        else:
            return torch.sqrt( (x-y)**2 + self.eps ).mean()





