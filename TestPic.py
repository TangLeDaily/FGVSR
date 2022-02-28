import argparse
import os
import random

import wandb
import numpy
import torch
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import nn
from PIL import Image, ImageFilter
from torchvision import transforms
from torch.nn import functional as F

from dataset_E import *
from util import *
from model import *
from srresnet import *
from rootmodel.EFVSR_M_TSALSTM_DSAdd_CellConv_test import *

parser = argparse.ArgumentParser(description="PyTorch Data_Pre")
parser.add_argument("--train_root_path", default='datasets/train/', type=str, help="train root path")
parser.add_argument("--test_root_path", default='datasets/test/REDS4/', type=str, help="test root path")
parser.add_argument("--cuda", default=True, action="store_true", help="use cuda?")
parser.add_argument("--frame", default=100, type=int, help="use cuda?")
parser.add_argument("--model_mark", default=0, type=int, help="which model to train? 0:default")
parser.add_argument("--resume", default='', type=str, help="path to latest checkpoint (default: none)")
parser.add_argument("--start_epoch", default=0, type=int, help="manual epoch number (useful on restarts)")
parser.add_argument("--batchSize", type=int, default=16, help="training batch size")  # default 16
parser.add_argument("--nEpochs", type=int, default=10000, help="number of epochs to train for")
parser.add_argument("--miniEpochs", type=int, default=0, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-5, help="Learning Rate. Default=1e-4")
parser.add_argument("--threads", type=int, default=8, help="number of threads for data loader to use")
parser.add_argument("--scale", type=int, default=4, help="Scale default:4x")
parser.add_argument("--loss", type=int, default=0, help="the loss function, default")
use_wandb = False
opt = parser.parse_args()
min_avr_loss = 99999999
save_flag = 0
epoch_avr_loss = 0
n_iter = 0
in_nc = 3
out_nc = 3

def main():
    global model, opt
    print(opt)

    print("===> Find Cuda")
    cuda = opt.cuda
    # torch.cuda.set_device(1)
    torch.cuda.set_device(0)
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    opt.seed = random.randint(1, 10000)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True

    print("===> Building model")
    if opt.model_mark == 0:
        model = EFVSR()
    criterion = nn.MSELoss()

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    print("===> Do Resume Or Skip")
    checkpoint = torch.load("checkpoints/over/TEST/model_epoch_0_psnr_27.3532.pth", map_location='cpu')
    model.load_state_dict(checkpoint.state_dict())
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.lo = 0
            model.load_state_dict(checkpoint.state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    print("===> Training")
    psnr = test_train_set(model, 333)



def test_train_set(this_model, epoch_num):
    dirname = "Teswt"
    miniDirname = "start_embedding_out_NoRB"
    print(" -- Start eval --")
    test_set = test_data_set(opt.test_root_path, "000/")
    test_loader = DataLoader(dataset=test_set, batch_size=1, num_workers=0)
    psnr = AverageMeter()
    with torch.no_grad():
        model = this_model
        if opt.cuda:
            model = model.cuda()
        model.eval()
        last_h = None
        last_c = None
        for iteration, batch in enumerate(test_loader, 1):
            input, target, I_D = batch
            if opt.cuda:
                input = input.cuda()
                target = target.cuda()
            out, last_h, last_c = model(input, last_h=last_h, last_c=last_c)
            # save_pic(out[0, :, :, :], "data/out/{}/out/".format(dirname), "{}_out.png".format(iteration))
            # save_pic(out_h[0, :, :, :], "data/out/{}/{}/".format(dirname, miniDirname), "{}_out.png".format(iteration))
            psnr.update(calc_psnr(out, target), len(out))
            # print(calc_psnr(out, target))
        if use_wandb:
            wandb.log({'epoch': epoch_num, 'psnr': psnr.avg})
        print("--->This--EFVSR_M_TSALSTM_DSAdd_SkipRB--epoch:{}--Avg--PSNR: {:.4f} dB--Root--PSNR: 24.11 dB".format(epoch_num,
                                                                                               psnr.avg))
    return psnr.avg


if __name__ == "__main__":
    main()
