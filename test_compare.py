import argparse
import os
import random

import dataset_E
import wandb
import numpy
import torch
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import nn
from PIL import Image, ImageFilter
from torchvision import transforms

from rootmodel.edvr import *
from util import *
from model import *
from srresnet import *
from rootmodel.SPVSR_New import *

def del_test_feat():
    if os.path.exists("data/feature/test_G.npy"):
        os.remove("data/feature/test_G.npy")

def del_train_feat(filename):
    for i in range(3):
        if os.path.exists("data/feature/" + filename + "_{}.npy".format(i)):
            os.remove("data/feature/" + filename + "_{}.npy".format(i))
def test_train_set():
    wandb.init(project="TEST", name="EDVR-OUR-compare-Vid4-calendar", entity="karledom")

    print("===> Find Cuda")
    if not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    seed = random.randint(1, 10000)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark = True

    print("===> Building model")
    modelA = SPVSR_GNew()
    modelB = EDVR()

    print("===> Do Resume Or Skip")
    checkpointA = torch.load("checkpoints/test/model_epoch_125_psnr_26.1414.pth", map_location='cuda:0')
    modelA.load_state_dict(checkpointA.state_dict())
    checkpointB = torch.load("checkpoints/test/model_videoID_TOTA_epoch_48.pth", map_location='cuda:0')
    modelB.load_state_dict(checkpointB.state_dict())

    print(" -- Start eval --")
    del_test_feat()
    test_set_A = dataset_E.test_data_set("datasets/test/REDS4/", "011/")
    test_loader_A = DataLoader(dataset=test_set_A, batch_size=1,  num_workers=1)
    test_set_B = dataset_E.test_data_set("datasets/test/REDS4/", "011/")
    test_loader_B = DataLoader(dataset=test_set_B, batch_size=1, num_workers=1)
    psnrA = AverageMeter()
    psnrB = AverageMeter()
    psnrRoot = AverageMeter()
    with torch.no_grad():
        modelA = modelA.cuda()
        modelB = modelB.cuda()
        modelA.eval()
        modelB.eval()

        i = 0
        del_train_feat("test_new_add_unet")
        for batchA, batchB in zip(test_loader_A, test_loader_B):
            ID = "test_new_add_unet"
            inputA, targetA, IDs = batchA
            inputB, targetB = batchB
            inputA = inputA.cuda()
            targetA = targetA.cuda()
            inputB = inputB.cuda()
            targetB = targetB.cuda()
            outA = modelA(inputA, ID)
            outB = modelB(inputB)
            save_pic(outA[0, :, :, :], "A/", "{}.png".format(i))
            save_pic(outB[0, :, :, :], "B/", "{}.png".format(i))
            i += 1

            psnr_our = calc_psnr(outA, targetA)
            psnr_edvr = calc_psnr(outB, targetB)
            psnr_Root = calc_psnr(F.interpolate(inputA, scale_factor=4, mode='bilinear', align_corners=False), targetA)

            psnrA.update(psnr_our)
            psnrB.update(psnr_edvr)
            psnrRoot.update(psnr_Root)

            wandb.log({'psnr_OUR_de': psnr_our-psnr_Root, 'psnr_EDVR_de': psnr_edvr-psnr_Root})
            # wandb.log({'OUR-EDVR': psnr_our - psnr_edvr})
        print("--->This---Our-Avg--PSNR: {:.4f} dB--EDVR-Avg--PSNR: {:.4f} dB--ROOT-Avg--PSNR: {:.4f} dB".format(psnrA.avg, psnrB.avg, psnrRoot.avg))

if __name__ == "__main__":
    test_train_set()