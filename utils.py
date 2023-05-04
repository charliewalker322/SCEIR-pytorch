import torchvision
import time


def save_TensorImg(img_tensor, path, nrow=1):
    torchvision.utils.save_image(img_tensor, path, nrow=nrow)


def write_config_to_file(config, file_path):
    """config should be a dict"""
    with open(file_path, 'w+') as f:
        f.write(time.asctime(time.localtime(time.time())) + '\n')
        for key, value in config.items():
            f.write(key+"       "+str(value) + '\n')
        f.close()


def write_metric_to_file(metric, file_path, opts, epoch):
    """metric should be a dict, file_path should exist"""
    if epoch == opts.eval_epoch:
        with open(file_path, "w+") as f:   # 重跑的话清空原来的内容
            f.close()
    with open(file_path, 'a+') as f:
        f.write(time.asctime(time.localtime(time.time())) + '\n')
        f.write("epoch       " + str(epoch) + '\n')
        for key, value in metric.items():       
            f.write(key + "     "+str(value) + '\n')
        f.write('\n')
        f.close()




