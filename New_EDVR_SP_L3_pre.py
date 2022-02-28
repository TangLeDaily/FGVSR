# coding=gbk
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


from dataset_New_L3 import *
from util import *
from rootmodel.EDVR_SP_L3FSR import *

use_wandb = False
Project_name = "SPVSR_SUP"
This_name = "EDVR_pre_SP_L3FSR"

parser = argparse.ArgumentParser(description="PyTorch Data_Pre")
parser.add_argument("--train_root_path", default='datasets/train/', type=str, help="train root path")
parser.add_argument("--test_root_path", default='datasets/test/', type=str, help="test root path")
parser.add_argument("--cuda", default=True, action="store_true", help="use cuda?")
parser.add_argument("--frame", default=100, type=int, help="use cuda?")
parser.add_argument("--start_epoch", default=0, type=int, help="manual epoch number (useful on restarts)")
parser.add_argument("--batchSize", type=int, default=16, help="training batch size")  # default 16
parser.add_argument("--nEpochs", type=int, default=10000, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.0004, help="Learning Rate. Default=1e-4")
parser.add_argument("--threads", type=int, default=16, help="number of threads for data loader to use")
opt = parser.parse_args()

def get_yu(model):
    kk = torch.load("checkpoints/over/TSALSTM_ATD/model_epoch_212_psnr_27.3702.pth", map_location='cpu')
    torch.save(kk.state_dict(), "checkpoints/test/New_130.pth")
    pretrained_dict = torch.load("checkpoints/test/New_130.pth")
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

def get_noTSA(model):
    model_dict = model.state_dict()
    checkpoint = torch.load("checkpoints/state/EDVR_M_x4_SR_REDS_official-32075921.pth",
                            map_location='cpu')
    checkpoint = checkpoint['params']
    new_state_dict = {}
    for k, v in checkpoint.items():
        if 'fusion' not in k:
            # if 'pcd_align' not in k:
            new_state_dict[k] = v
    new_state_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
    model_dict.update(new_state_dict)  # 利用预训练模型的参数，更新模型
    model.load_state_dict(model_dict)
    return model

def main():
    global model, opt
    if use_wandb:
        wandb.init(project=Project_name, name=This_name, entity="karledom")
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
    # cudnn.benchmark = True

    print("===> Loading datasets")

    test_set_1 = test_data_set(opt.test_root_path, "000/")
    test_loader_1 = DataLoader(dataset=test_set_1, batch_size=1, num_workers=0)
    test_set_2 = test_data_set(opt.test_root_path, "011/")
    test_loader_2 = DataLoader(dataset=test_set_2, batch_size=1, num_workers=0)
    test_set_3 = test_data_set(opt.test_root_path, "015/")
    test_loader_3 = DataLoader(dataset=test_set_3, batch_size=1, num_workers=0)
    test_set_4 = test_data_set(opt.test_root_path, "020/")
    test_loader_4 = DataLoader(dataset=test_set_4, batch_size=1, num_workers=0)

    test_loader_sum = []
    test_loader_sum.append(test_loader_1)
    test_loader_sum.append(test_loader_2)
    test_loader_sum.append(test_loader_3)
    test_loader_sum.append(test_loader_4)

    print("===> Building model")
    model = EDVR()
    criterion = nn.MSELoss()

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    print("===> Do Resume Or Skip")
    model = get_noTSA(model)
    # state_dict = torch.load('checkpoints/state/EDVR_M_x4_SR_REDS_official-32075921.pth', map_location='cpu')
    # state_dict = state_dict['params']
    # model.load_state_dict(state_dict)
    # state_dict = torch.load('model_name.pth', map_location='cpu')
    # model.load_state_dict(state_dict['model'])
    # model = get_yu(model)

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))

    print("===> Training")
    last_psnr = 0
    Now_best_psnr = 0
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(optimizer, model, criterion, epoch)
        psnr = test_train_set(model, test_loader_sum, epoch)
        save_checkpoint(model, psnr, epoch, optimizer.param_groups[0]["lr"])
        if psnr > Now_best_psnr:
            Now_best_psnr = psnr
        if epoch % 5 == 0:
          if Now_best_psnr <= last_psnr:
            for p in optimizer.param_groups:
                p['lr'] *= 0.8
          last_psnr = Now_best_psnr

def train(optimizer, model, criterion, epoch):
    global opt
    train_set = train_data_set(opt.train_root_path, batchsize=opt.batchSize)
    train_dataloader = DataLoader(dataset=train_set, batch_size=opt.batchSize, num_workers=opt.threads,
                                      drop_last=True)
    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()
    avg_loss = AverageMeter()
    for iteration, batch in enumerate(train_dataloader):
        input, target = batch
        if iteration % opt.frame == 0:
            last_h = None
            last_c = None
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()
        out, last_h, last_c = model(input, last_h=last_h, last_c=last_c)
        loss = criterion(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss.update(loss.item())
        if iteration % 50 == 0:
            if use_wandb:
                wandb.log({'epoch': epoch, 'iter_loss': avg_loss.avg})
            print('epoch_iter_{}_loss is {:.10f}'.format(iteration, avg_loss.avg))


def save_checkpoint(model, psnr, epoch, lr):
    global opt

    model_folder = "checkpoints/{}/".format(This_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    model_out_path = model_folder + "epoch_{}_psnr_{:.4f}_lr_{}.pth".format(epoch, psnr, lr)
    torch.save({'model': model.state_dict()}, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def test_train_set(this_model, test_loader_sum, epoch_num):
    print(" -- Start eval --")
    psnr_sum = 0
    ssim_sum = 0
    with torch.no_grad():
        for iii, test_loader in enumerate(test_loader_sum):
            psnr = AverageMeter()
            ssim = AverageMeter()
            model = this_model
            if opt.cuda:
                model = model.cuda()
            model.eval()
            last_h = None
            last_c = None
            for iteration, batch in enumerate(test_loader, 1):
                input, target = batch
                if opt.cuda:
                    input = input.cuda()
                    target = target.cuda()
                out, last_h, last_c = model(input, last_h=last_h, last_c=last_c)
                psnr.update(calc_psnr(out, target), len(out))
                ssim.update(cal_ssim_tensor(out[0, :, :, :], target[0, :, :, :]), len(out))
            if use_wandb:
                wandb.log({'psnr{}'.format(iii+1): psnr.avg, 'ssim{}'.format(iii+1): ssim.avg})
            print("--->This--{}--epoch:{}--Avg--PSNR: {:.4f} dB -- SSIM :{}--Dir: {}".format(This_name, epoch_num, psnr.avg, ssim.avg, iii+1))
            psnr_sum += psnr.avg
            ssim_sum += ssim.avg
    print(" -- Sum PSNR: {:.4f} -- ".format(psnr_sum/4.))
    if use_wandb:
        wandb.log({'epoch': epoch_num, 'psnr_sum': psnr_sum/4.})
    print(" -- Sum SSIM: {:.4f} -- ".format(ssim_sum / 4.))
    if use_wandb:
        wandb.log({'epoch': epoch_num, 'ssim_sum': ssim_sum / 4.})
    return psnr_sum/4.


if __name__ == "__main__":
    main()
