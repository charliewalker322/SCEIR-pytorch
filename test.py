import time
import os
import argparse
from utils import save_TensorImg
from tqdm import tqdm
from toolkit import *
from train import SCEIR_Model


def test(model, opts):
    model.load_checkpoint(opts.load_model, opts.device)
    test_dataloader = TestLoading(opts)

    # create save dirs
    opts.test_output = os.path.join(opts.test_output, time.strftime('%Y-%m-%d %H_%M'))
    save_path = os.path.join(opts.test_output, 'output')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    contrast_path = os.path.join(opts.test_output, 'contrast')
    if not os.path.exists(contrast_path):
        os.makedirs(contrast_path)

    global_path = os.path.join(opts.test_output, 'global')
    if not os.path.exists(global_path):
        os.makedirs(global_path)

    runtime = 0
    length = len(test_dataloader)-1
    print('------> Enhance start!')
    for i, batch in enumerate(tqdm(test_dataloader)):
        data = getData_hsv(batch, opts, mode='test')

        model.eval()
        with torch.no_grad():
            t1 = time.time()
            output, ain_out, pre_std, pre_mean = model(data)
            t2 = time.time()
            # runtime += t2 - t1

            # for 3090 or A100, the speed of first processed image will be much lower than the following ones
            if i>0:
                runtime += t2-t1

            output = output.clamp(0, 1)
            ain_out = ain_out.clamp(0, 1)

            # save extra
            if opts.save_extra:
                # save contrast img
                contrast = torch.cat([data['raw'], ain_out,  output])
                output_path = os.path.join(contrast_path, 'contrast_' + data['name'] + '.png')
                save_TensorImg(contrast, path=output_path, nrow=contrast.size(0))

                # save global enhanced img
                ain_output_path = os.path.join(global_path, 'global_' + data['name'] + '.png')
                save_TensorImg(ain_out, path=ain_output_path)

            # save output
            output_path = os.path.join(save_path, 'SCEIR_' + data['name'] + '.png')
            save_TensorImg(output, path=output_path)


    print(f'------> Runtime : {runtime/length:.4f}')
    print('------> Enhance done!')


if __name__ == "__main__":
    torch.backends.cudnn.enabled = False

    parser = argparse.ArgumentParser(description='Configure')
    # training config
    parser.add_argument('--gpu', type=str, default='0')   # set -1 to use cpu
    parser.add_argument('--downsample', type=float, default=1.0)    # whether to use downsample in the range (0,1]
    parser.add_argument('--load_model', type=str, default='./checkpoint/model.pthep145')   # checkpoint path
    parser.add_argument('--save_extra', action='store_true', default=False) # whether to save contrast/global images
    parser.add_argument('--test_input', type=str, default='./test_input')   # input path
    parser.add_argument('--test_output', type=str, default='./test_output') # output path

    opts = parser.parse_args()

    if torch.cuda.is_available() and int(opts.gpu) >= 0:
        opts.device = f'cuda:{opts.gpu}'
    else:
        opts.device = 'cpu'

    # print settings
    print('<-------settings-------->')
    for k, v in vars(opts).items():
        print('--',k, v)

    model = SCEIR_Model(opts, mode='test').to(opts.device)
    test(model, opts)
