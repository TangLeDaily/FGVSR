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

from dataset_sr import *
from util import *
from model import *
from srresnet import *
from rootmodel.edvr import *

modelname = "EDVR_M"
version = "-V1"
# model = Network()

parser = argparse.ArgumentParser(description="PyTorch Data_Pre")
parser.add_argument("--train_root_path", default='datasets/train/', type=str, help="train root path")
parser.add_argument("--test_root_path", default='datasets/test/REDS4/', type=str, help="test root path")
parser.add_argument("--cuda", default=True, action="store_true", help="use cuda?")
parser.add_argument("--model_mark", default=0, type=int, help="which model to train? 0:default")
parser.add_argument("--resume", default='', type=str, help="path to latest checkpoint (default: none)")
parser.add_argument("--start_epoch", default=0, type=int, help="manual epoch number (useful on restarts)")
parser.add_argument("--batchSize", type=int, default=16, help="training batch size")  # default 16
parser.add_argument("--nEpochs", type=int, default=10000, help="number of epochs to train for")
parser.add_argument("--miniEpochs", type=int, default=0, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate. Default=1e-4")
parser.add_argument("--threads", type=int, default=8, help="number of threads for data loader to use")
parser.add_argument("--scale", type=int, default=4, help="Scale default:4x")
parser.add_argument("--loss", type=int, default=0, help="the loss function, default")
use_wandb = True
opt = parser.parse_args()
min_avr_loss = 99999999
save_flag = 0
epoch_avr_loss = 0
n_iter = 0
in_nc = 3
out_nc = 3
temp = 0
def get_yu(model):
    kk = torch.load("checkpoints/over/edvr_mini_olddataset/model_videoID_TOTA_epoch_48.pth", map_location='cpu')
    torch.save(kk.state_dict(), "checkpoints/state/New_48.pth")
    pretrained_dict = torch.load("checkpoints/state/New_48.pth")
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)  # 利用预训练模型的参数，更新模型
    model.load_state_dict(model_dict)
    return model

def get_yu2(model):
    kk = torch.load("checkpoints/default/net_g_300000.pth")
    model_dict = model.state_dict()
    kk = {k: v for k, v in kk.items() if k in model_dict}
    model_dict.update(kk)  # 利用预训练模型的参数，更新模型
    model.load_state_dict(model_dict)
    return model

def main():
    global model, opt
    if use_wandb:
        wandb.init(project="duibi", name=modelname + version, entity="karledom")
    print(opt)

    print("===> Find Cuda")
    cuda = opt.cuda
    torch.cuda.set_device(0)
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    opt.seed = random.randint(1, 10000)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True

    print("===> Building model")
    model = EDVR()
    # 加载模型
    criterion = nn.MSELoss()

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    print("===> Do Resume Or Skip")
    # model = get_yu(model)
    # checkpoint = torch.load("checkpoints/EDVR_L/model_videoID_239_epoch_14.pth")
    # model.load_state_dict(checkpoint.state_dict())
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
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train_total(optimizer, model, criterion, epoch)


def train_total(optimizer, model, criterion, epoch):
    global temp
    path_lis = get_path(opt.train_root_path)  # 000/  001/  002/

    for video_id in path_lis:
        train_set = train_data_set(opt.train_root_path, video_id)
        train_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize, shuffle=True, num_workers=opt.threads,
                                  drop_last=True)
        for miniepoch in range(0, opt.miniEpochs + 1):
            train_mini(train_loader, optimizer, model, criterion, miniepoch, video_id)
    psnr = test_train_set(model, epoch, "NoVid")
    print("________SAVE______________")
    save_checkpoint(model, video_id, epoch, psnr)
    # test_total_set(model, epoch)


def train_mini(train_loader, optimizer, model, criterion, epoch, video_id):
    global min_avr_loss
    global save_flag
    global epoch_avr_loss
    global n_iter

    avr_loss = 0

    print("video_id={}, Epoch={}, lr={}".format(video_id[:-1], epoch, optimizer.param_groups[0]["lr"]))
    model.train()
    for iteration, batch in enumerate(train_loader, 1):
        n_iter = iteration
        input, target = batch
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()
        out = model(input)
        loss = criterion(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avr_loss += loss.item()
    avr_loss = avr_loss / len(train_loader)
    epoch_avr_loss = avr_loss
    if use_wandb and avr_loss < 0.01:
        wandb.log({'video_id': video_id[:-1], 'epoch': epoch, 'loss': avr_loss})
    print('model_{}_epoch_avr_loss is {:.10f}'.format(modelname, epoch_avr_loss))


def save_checkpoint(model, videoid, epoch, psnr):
    global min_avr_loss
    global save_flag
    global opt
    model_folder = "checkpoints/" + modelname + "/"

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    model_out_path = model_folder + "model_videoID_{}_epoch_{}_psnr:{:.4f}.pth".format(videoid[:-1], epoch, psnr)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def test_train_set(this_model, epoch_num, videoid):
    print(" -- Start eval --")
    test_set = test_data_set(opt.test_root_path, "000/")
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=True, num_workers=opt.threads)
    psnr = AverageMeter()
    with torch.no_grad():
        model = this_model
        if opt.cuda:
            model = model.cuda()
        model.eval()
        for iteration, batch in enumerate(test_loader, 1):
            input, target = batch
            if opt.cuda:
                input = input.cuda()
                target = target.cuda()
            out = model(input)
            psnr.update(calc_psnr(out, target), len(out))
        if use_wandb:
            wandb.log({'epoch': epoch_num, 'psnr': psnr.avg})
        print("--->This--epoch:{}--videoid:{}--Avg--PSNR: {:.4f} dB--Root--PSNR: 24.11 dB".format(epoch_num, videoid,
                                                                                                  psnr.avg))
    return psnr.avg


if __name__ == "__main__":
    main()
