import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import glob
from torch.nn.functional import interpolate
import os


class TrainLoading(Dataset):
    def __init__(self, opts):
        self.opts = opts
        self.toTensor = transforms.ToTensor()
        self.files_low = sorted( glob.glob(os.path.join(opts.train_raw, '*.*g')) )
        self.files_high = sorted( glob.glob(os.path.join(opts.train_ref, '*.*g')) )

    def __getitem__(self, index):
        patch_low = self.toTensor(Image.open(self.files_low[index % len(self.files_low)]))
        patch_high = self.toTensor(Image.open(self.files_high[index % len(self.files_high)]))

        # resize
        if self.opts.size>0:
            patch_low = interpolate(patch_low.unsqueeze(0), (self.opts.size, self.opts.size), mode='bilinear').squeeze(0)
            patch_high = interpolate(patch_high.unsqueeze(0), (self.opts.size, self.opts.size), mode='bilinear').squeeze(0)

        return {
            'raw': patch_low,
            'ref': patch_high,
            'name': os.path.basename(self.files_low[index]).split('.')[0]
        }

    def __len__(self):
        return min(len(self.files_low), len(self.files_high))


class EvalLoading():
    def __init__(self, opts, resize=-1):
        self.opts = opts
        self.resize = resize
        self.toTensor = transforms.ToTensor()
        self.files_low = sorted( glob.glob(os.path.join(opts.eval_raw, '*.*g')) )
        self.files_high = sorted( glob.glob(os.path.join(opts.eval_ref, '*.*g')) )

    def __getitem__(self, index):
        patch_low = self.toTensor(Image.open(self.files_low[index])).unsqueeze(0)
        patch_high = self.toTensor(Image.open(self.files_high[index])).unsqueeze(0)

        return {
            'raw': patch_low,
            'ref': patch_high,
            'name': os.path.basename(self.files_low[index]).split('.')[0]
        }

    def __len__(self):
        return min(len(self.files_low), len(self.files_high))


class TestLoading():
    def __init__(self, opts):
        self.opts = opts
        self.toTensor = transforms.ToTensor()
        self.files_raw = sorted(glob.glob(os.path.join(opts.test_input, '*.*g')))

    def __getitem__(self, index):
        file_name = self.files_raw[index]
        low = self.toTensor(Image.open(file_name)).unsqueeze(0)

        # downsample
        if self.opts.downsample>1.0 or self.opts.downsample<1.0:
            low = interpolate(low, scale_factor=self.opts.downsample, mode='bilinear', align_corners=True)

        return {'name': os.path.basename(file_name).split('.')[0],
                'raw': low}

    def __len__(self):
        return len(self.files_raw)


def getData_hsv(data:dict, opts, mode='train'):

    data['raw'] = data['raw'].to(opts.device)
    data['raw_v'] = data['raw'].max(dim=1, keepdim=True)[0]

    if mode=='train':
        data['ref'] = data['ref'].to(opts.device)

    return data


def cat1(*xlist):
    if type(xlist[0]) is tuple:
        xlist = xlist[0]
    return torch.cat(xlist,dim=1).to(xlist[0])


def loss_eff_create(loss:str):
    temp_loss = loss.split('-')
    losses = {}
    for item in temp_loss:
        if '*' in item:
            co = item.split('*')
            losses[co[1]] = float(co[0])
        elif '+' in item:
            co = item.split('+')
            losses[co[1]] = float(co[0])
        else:
            losses[item] = 1.0
    return losses, list(losses.keys())
