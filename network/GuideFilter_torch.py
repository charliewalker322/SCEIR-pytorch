import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
def diff_x(input, r):
    assert input.dim() == 4

    left   = input[:, :,         r:2 * r + 1]
    middle = input[:, :, 2 * r + 1:         ] - input[:, :,           :-2 * r - 1]
    right  = input[:, :,        -1:         ] - input[:, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=2)

    return output

def diff_y(input, r):
    assert input.dim() == 4

    left   = input[:, :, :,         r:2 * r + 1]
    middle = input[:, :, :, 2 * r + 1:         ] - input[:, :, :,           :-2 * r - 1]
    right  = input[:, :, :,        -1:         ] - input[:, :, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=3)

    return output

class BoxFilter(nn.Module):
    def __init__(self, r):
        super(BoxFilter, self).__init__()

        self.r = r

    def forward(self, x):
        assert x.dim() == 4

        return diff_y(diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)

class GuidedFilter(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(GuidedFilter, self).__init__()
        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)

    def forward(self, x, y, r=0, eps=0):
        # 如果参数有效，使用指定参数
        if r>0 and eps>0:
            boxfilter = BoxFilter(r)
        else:
            boxfilter = self.boxfilter
            eps = self.eps
            r = self.r

        n_x, c_x, h_x, w_x = x.size()
        n_y, c_y, h_y, w_y = y.size()

        assert n_x == n_y
        assert c_x == 1 or c_x == c_y
        assert h_x == h_y and w_x == w_y
        assert h_x > 2 * r + 1 and w_x > 2 * r + 1
        # N
        N = boxfilter(Variable(x.data.new().resize_((1, 1, h_x, w_x)).fill_(1.0)))
        # mean_x
        mean_x = boxfilter(x) / N
        # mean_y
        mean_y = boxfilter(y) / N
        # cov_xy
        cov_xy = boxfilter(x * y) / N - mean_x * mean_y
        # var_x
        var_x = boxfilter(x * x) / N - mean_x * mean_x

        # A
        A = cov_xy / (var_x + eps)
        # b
        b = mean_y - A * mean_x

        # mean_A; mean_b
        mean_A = boxfilter(A) / N
        mean_b = boxfilter(b) / N

        return mean_A * x + mean_b
    