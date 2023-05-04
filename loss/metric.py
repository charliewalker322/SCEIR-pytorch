import torch
import torch.nn as nn
from math import exp
import torch.nn.functional as F
from torch.autograd import Variable


class SSIM(nn.Module):
    def __init__(self):
        super().__init__()
        self.window_size = 11
        
    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(
            _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        )
        return window
    
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2))
            for x in range(window_size)
        ]
        )
        return gauss / gauss.sum()

    def _ssim(self, im1, im2, window, window_size, channel, average):
        mu1 = F.conv2d(im1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(im2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            F.conv2d(im1 * im1, window, padding=window_size // 2, groups=channel)
            - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(im2 * im2, window, padding=window_size // 2, groups=channel)
            - mu2_sq
        )
        sigma12 = (
            F.conv2d(im1 * im2, window, padding=window_size // 2, groups=channel)
            - mu1_mu2
        )

        C1 = (0.01) ** 2
        C2 = (0.03) ** 2
        up = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
        down = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        ssim_map = up / down

        if average:
            return ssim_map.mean()
        else:
            return ssim_map

    def forward(self, im1, im2, average=True):
        channel = im1.size(1)
        window = self.create_window(self.window_size, channel)
        if im1.is_cuda:
            window = window.cuda(im1.get_device())
        window = window.type_as(im1)
        return self._ssim(im1, im2, window, self.window_size, channel, average)

class PSNR(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, im1, im2):
        #bs = im1.size(0)
        #mse_err = (im1 - im2).pow(2).sum(dim=1).view(bs, -1).mean(dim=1)
        #psnr = 10 * (1 / mse_err).log10()
        a = torch.log10(1. * 1. / nn.MSELoss()(im1, im2)) * 10
        psnr = torch.clamp(a, 0., 99.99)
        return psnr.mean()

