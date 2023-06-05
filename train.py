import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
import os
import argparse

from torch.utils.data import DataLoader
from utils import save_TensorImg, write_metric_to_file, write_config_to_file
from tqdm import tqdm
from loss.loss_function import perceptual_loss, Charbonnier_loss
from loss.metric import SSIM, PSNR
from toolkit import *
from network.GuideFilter_torch import GuidedFilter
from network.water_network import MSVector, Plain5
import itertools


class SCEIR_Model(nn.Module):
    def __init__(self, opts, mode='train'):
        super().__init__()
        self.opts = opts

        self.SCE_model = MSVector(std_ch=3, mean_ch=3, size=512, fea=32)
        self.LRN_model = Plain5(in_ch=4, out_ch=3, finnal_activate=False, channel_attention=True, fea=[64,64,64,64])

        self.guide_filter = GuidedFilter(r=10, eps=1)

        if mode=='train':
            #  losses
            self.psnr = PSNR()
            self.ssim = SSIM()
            self.vgg = perceptual_loss(opts.device)
            self.char_loss = Charbonnier_loss()

            para = itertools.chain(self.SCE_model.parameters(), self.LRN_model.parameters())
            self.optimizer = torch.optim.Adam(
                para,
                lr=opts.lr,
                betas=(0.9, 0.999),
                weight_decay=opts.weight_decay,
            )


    def forward(self, batch):
        raw = batch['raw']
        raw_std, raw_mean = torch.std_mean(raw, dim=(2, 3), keepdim=True, unbiased=False)

        # gray-ratio   ratio =  mean channel / mean red channel   # Inspired by the Gray World
        pre_std, pre_mean = self.SCE_model(raw)
        ratio = batch['raw'].mean(dim=(1,2,3)).reshape(-1,1,1) / batch['raw'][:, 0].mean(dim=(1, 2), keepdim=True)
        pre_mean[:,0] *= ratio
        ain_out = pre_std * (raw - raw_mean) / raw_std + pre_mean

        # concat gf_v
        gf_v = batch['raw_v']*2 - self.guide_filter(batch['raw_v'], batch['raw_v'])
        out = self.LRN_model(cat1(ain_out, gf_v)) + ain_out    # res

        return out, ain_out, pre_std, pre_mean

    def save_checkpoint(self, model_path):
        state_dict = {
            'restore_model': self.SCE_model.state_dict(),
            'post_model': self.LRN_model.state_dict()
        }
        checkpoint_state = {
            "state_dict": state_dict,
            "epoch": self.epoch + 1,
            "opts": self.opts,
        }
        torch.save(checkpoint_state, model_path)
        print("---saving to: ", model_path)

    def load_checkpoint(self, model_path, device):
        checkpoint = torch.load(model_path, map_location=device)
        self.SCE_model.load_state_dict(checkpoint['state_dict']['restore_model'])
        self.LRN_model.load_state_dict(checkpoint['state_dict']['post_model'])
        print('------------------Successfully load model -> {}'.format(model_path))



def eval(model, opts, epoch, dataloader):
    print("Evaling the " + str(epoch) + " epoch")
    metric = {}

    for iteration, data in enumerate(tqdm(dataloader)):
        data = getData_hsv(data, opts, mode='train')

        model.eval()
        with torch.no_grad():
            output, ain_out, pre_std, pre_mean = model( data )

            output = output.clamp(0, 1)
            ain_out = ain_out.clamp(0, 1)

            metrics = {
                       'ssim': model.ssim( output, data['ref'] ),
                       'psnr': model.psnr( output, data['ref'] ),
                        'AIN_ssim': model.ssim(ain_out, data['ref']),
                        'AIN_psnr': model.psnr(ain_out, data['ref']),
                       }

        model.train()

        ################## save contrast-output-img
        img = torch.cat( [ data['raw'], ain_out, output, data['ref'] ], dim=0 )
        con_path = os.path.join(opts.saving_eval_dir, "contrast")
        save_TensorImg(img, path=os.path.join(con_path, "{}_epoch{}.png".format(data['name'], epoch)), nrow=img.size(0))

        # save output
        img_path = os.path.join(opts.saving_eval_dir, "output")
        output_path = os.path.join(img_path, "{}_epoch{}.png".format(data['name'], epoch))
        save_TensorImg(output, path=output_path)


        for key in metrics.keys():
            if key not in metric.keys():
                metric[key] = metrics[key]
            else:
                metric[key] = metric[key] + metrics[key]

    print(" =========================================== > evaling done!")
    return {l: (metric[l] / float(iteration + 1)) for l in metric.keys()}


def train(model, opts, epoch, dataloader ):
    print("Training the " + str(epoch) + " epoch ...")
    loss = {}

    model.train()
    for iteration, data in enumerate(dataloader):
        data = getData_hsv(data, opts, mode='train')

        model.optimizer.zero_grad()

        output, ain_out, pre_std, pre_mean = model(data)

        # metric
        losses = {}
        losses['ssim'] = model.ssim(output.clamp(0,1), data['ref'])
        losses['psnr'] = model.psnr(output.clamp(0,1), data['ref'])


        ##  loss ###
        ref_std, ref_mean = torch.std_mean(data['ref'], dim=(2, 3), unbiased=False)
        losses['Char_loss'] = model.char_loss(pre_std.mean(dim=(2, 3)), ref_std) \
                                  + model.char_loss(pre_mean.mean(dim=(2, 3)), ref_mean)
        losses['MSE_loss'] = mse_loss(output, data['ref'])
        losses['SSIM_loss'] = 1 - model.ssim(output, data['ref'])
        losses['VGG_loss'] = 0.1 * model.vgg(output, data['ref'])

        total_loss = losses['Char_loss'] + losses['MSE_loss'] + losses['SSIM_loss'] + losses['VGG_loss']
        losses['total_loss'] = total_loss
        total_loss.mean().backward()
        model.optimizer.step()


        for key in losses.keys():
            if key in loss.keys():
                loss[key] = losses[key].detach().mean().item() + loss[key]
            else:
                loss[key] = losses[key].detach().mean().item()

        str_to_print = "Train: Epoch {}: {}/{} with ".format(epoch, iteration, len(dataloader))
        for key in loss.keys():
            str_to_print += " %s : %0.6f | " % (key, loss[key] / float(iteration + 1))
        print(str_to_print)


def run(model, opts):

    # load datasets
    train_dataset = TrainLoading(opts)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4)

    eval_dataloader = EvalLoading(opts)


    ########## save path setting ######################
    model_path = os.path.join( opts.model_path, 'model', opts.loss)
    opts.saving_eval_dir = os.path.join(opts.saving_eval_dir, 'eval', opts.loss)

    if not os.path.exists(os.path.join(opts.saving_eval_dir, "contrast")):
        os.makedirs(os.path.join(opts.saving_eval_dir, "contrast"))
    if not os.path.exists(os.path.join(opts.saving_eval_dir, "output")):
        os.makedirs(os.path.join(opts.saving_eval_dir, "output"))

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    opts.model_path = os.path.join( model_path , opts.loss + '_model.pth')

    ##################################################

    loss_settings, loss_options = loss_eff_create(opts.loss)
    print('loss_settings: ', loss_settings)

    model.train()
    for epoch in range(1, opts.epoch+1):
        model.epoch = epoch
        train(model=model, opts=opts, epoch=epoch, dataloader=train_dataloader)

        if ( epoch % opts.eval_epoch == 0 ) or epoch==5 :
            model.save_checkpoint( opts.model_path + "ep%d" % epoch)
            metric = eval(model=model, opts=opts, epoch=epoch, dataloader=eval_dataloader)
            write_metric_to_file(metric, os.path.join(opts.saving_eval_dir, "metric.txt"), opts, epoch)
            write_config_to_file(vars(opts), os.path.join(opts.saving_eval_dir, "config.txt"))




if __name__ == "__main__":
    torch.backends.cudnn.enabled = False

    parser = argparse.ArgumentParser(description='Configure')
    # training config
    parser.add_argument('--gpu', type=str, default='0')   # set -1 to use cpu
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--size', type=int, default=512)    # size of input images
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--eval_epoch', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=150)
    parser.add_argument('--loss', type=str, default='char-L2-SSIM-0.1+VGG')

    # dataset path
    parser.add_argument('--train_raw', type=str,
                        default="./dataset/UIEBD/train/raw")
    parser.add_argument('--train_ref', type=str,
                        default="./dataset/UIEBD/train/ref")

    parser.add_argument('--eval_raw', type=str,
                        default="./dataset/UIEBD/eval/raw")
    parser.add_argument('--eval_ref', type=str,
                        default="./dataset/UIEBD/eval/ref")

    # checkpoints config
    parser.add_argument('--model_path', type=str, default="./SCEIR_train")
    parser.add_argument('--saving_eval_dir', type=str, default="./SCEIR_train")


    opts = parser.parse_args()

    if torch.cuda.is_available() and int(opts.gpu) >= 0:
        opts.device = f'cuda:{opts.gpu}'
    else:
        opts.device = 'cpu'

    # print settings
    print('<-------settings-------->')
    for k, v in vars(opts).items():
        print('--',k, v)

    model = SCEIR_Model(opts).to(opts.device)
    run(model, opts)