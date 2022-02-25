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

from rootmodel.PFNL import *
from util import *


def del_test_feat():
    if os.path.exists("data/feature/test_G.npy"):
        os.remove("data/feature/test_G.npy")

def del_train_feat(filename):
    for i in range(3):
        if os.path.exists("data/feature/" + filename + "_{}.npy".format(i)):
            os.remove("data/feature/" + filename + "_{}.npy".format(i))
def test_train_set():
    # wandb.init(project="TEST", name="OUR-EFVSR_L_OnlypcdOneLSTM", entity="karledom")

    print("===> Find Cuda")
    if not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    seed = random.randint(1, 10000)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark = True

    print("===> Building model")
    model = PFNL()

    print("===> Do Resume Or Skip")
    checkpoint = torch.load("checkpoints/over/PFNL_26d85/model_epoch_24_psnr_26.8550.pth", map_location='cuda:0')
    model.load_state_dict(checkpoint.state_dict())

    print(" -- Start eval --")

    # print("JK-000")
    # del_train_feat("test_EF")
    # test_set = dataset_E.test_data_set("datasets/test/Videoget/", "000/")
    # test_loader = DataLoader(dataset=test_set, batch_size=1, num_workers=1)
    # psnrZA = AverageMeter()
    # SSIMZA = AverageMeter()
    # psnrRootZA = AverageMeter()
    # with torch.no_grad():
    #     model = model.cuda()
    #     model.eval()
    #     for batch in test_loader:
    #         ID = "test_EF"
    #         input, target, tID = batch
    #         input = input.cuda()
    #         target = target.cuda()
    #         out = model(input)
    #         SSIM = compute_ssim_tensor(out[0, :, :, :], target[0, :, :, :])
    #         psnr_our = calc_psnr(out, target)
    #         psnr_Root = calc_psnr(
    #             F.interpolate(input[:, 0, :, :, :], scale_factor=4, mode='bilinear', align_corners=False), target)
    #         psnrZA.update(psnr_our)
    #         SSIMZA.update(SSIM)
    #         psnrRootZA.update(psnr_Root)
    #     print("--->This---Our-Avg--PSNR: {:.4f} dB---ROOT-Avg--PSNR: {:.4f} dB-- SSIM:{:.4f}".format(psnrZA.avg,
    #                                                                                                  psnrRootZA.avg,
    #                                                                                                  SSIMZA.avg))
    #
    # print("JK-001")
    # del_train_feat("test_EF")
    # test_set = dataset_E.test_data_set("datasets/test/Videoget/", "001/")
    # test_loader = DataLoader(dataset=test_set, batch_size=1, num_workers=1)
    # psnrZB = AverageMeter()
    # SSIMZB = AverageMeter()
    # psnrRootZB = AverageMeter()
    # with torch.no_grad():
    #     model = model.cuda()
    #     model.eval()
    #     for batch in test_loader:
    #         ID = "test_EF"
    #         input, target, tID = batch
    #         input = input.cuda()
    #         target = target.cuda()
    #         out = model(input)
    #         SSIM = compute_ssim_tensor(out[0, :, :, :], target[0, :, :, :])
    #         psnr_our = calc_psnr(out, target)
    #         psnr_Root = calc_psnr(
    #             F.interpolate(input[:, 0, :, :, :], scale_factor=4, mode='bilinear', align_corners=False), target)
    #         psnrZB.update(psnr_our)
    #         SSIMZB.update(SSIM)
    #         psnrRootZB.update(psnr_Root)
    #     print("--->This---Our-Avg--PSNR: {:.4f} dB---ROOT-Avg--PSNR: {:.4f} dB-- SSIM:{:.4f}".format(psnrZB.avg,
    #                                                                                                  psnrRootZB.avg,
    #                                                                                                  SSIMZB.avg))


    print("REDS4-clip_000")
    del_train_feat("test_EF")
    test_set = dataset_E.test_data_set("datasets/test/REDS4/", "000/")
    test_loader = DataLoader(dataset=test_set, batch_size=1,  num_workers=1)
    psnrA = AverageMeter()
    SSIMA = AverageMeter()
    psnrRootA = AverageMeter()
    with torch.no_grad():
        model = model.cuda()
        model.eval()
        for batch in test_loader:
            ID = "test_EF"
            input, target, tID = batch
            input = input.cuda()
            target = target.cuda()
            out = model(input)
            SSIM = compute_ssim_tensor(out[0,:,:,:], target[0,:,:,:])
            psnr_our = calc_psnr(out, target)
            psnr_Root = calc_psnr(F.interpolate(input[:,0,:,:,:], scale_factor=4, mode='bilinear', align_corners=False), target)
            psnrA.update(psnr_our)
            SSIMA.update(SSIM)
            psnrRootA.update(psnr_Root)
        print("--->This---Our-Avg--PSNR: {:.4f} dB---ROOT-Avg--PSNR: {:.4f} dB-- SSIM:{:.4f}".format(psnrA.avg, psnrRootA.avg, SSIMA.avg))



    print("REDS4-clip_011")
    del_train_feat("test_EF")
    test_set = dataset_E.test_data_set("datasets/test/REDS4/", "011/")
    test_loader = DataLoader(dataset=test_set, batch_size=1, num_workers=1)
    psnrB = AverageMeter()
    SSIMB = AverageMeter()
    psnrRootB = AverageMeter()
    with torch.no_grad():
        model = model.cuda()
        model.eval()
        for batch in test_loader:
            ID = "test_EF"
            input, target, tID = batch
            input = input.cuda()
            target = target.cuda()
            out = model(input)
            SSIM = compute_ssim_tensor(out[0,:,:,:], target[0,:,:,:])
            psnr_our = calc_psnr(out, target)
            psnr_Root = calc_psnr(
                F.interpolate(input[:, 0, :, :, :], scale_factor=4, mode='bilinear', align_corners=False), target)
            psnrB.update(psnr_our)
            SSIMB.update(SSIM)
            psnrRootB.update(psnr_Root)
        print("--->This---Our-Avg--PSNR: {:.4f} dB---ROOT-Avg--PSNR: {:.4f} dB-- SSIM:{:.4f}".format(psnrB.avg, psnrRootB.avg, SSIMB.avg))

    print("REDS4-clip_015")
    del_train_feat("test_EF")
    test_set = dataset_E.test_data_set("datasets/test/REDS4/", "015/")
    test_loader = DataLoader(dataset=test_set, batch_size=1, num_workers=1)
    psnrC = AverageMeter()
    SSIMC = AverageMeter()
    psnrRootC = AverageMeter()
    with torch.no_grad():
        model = model.cuda()
        model.eval()
        for batch in test_loader:
            ID = "test_EF"
            input, target, tID = batch
            input = input.cuda()
            target = target.cuda()
            out = model(input)
            SSIM = compute_ssim_tensor(out[0,:,:,:], target[0,:,:,:])
            psnr_our = calc_psnr(out, target)
            psnr_Root = calc_psnr(
                F.interpolate(input[:, 0, :, :, :], scale_factor=4, mode='bilinear', align_corners=False), target)
            psnrC.update(psnr_our)
            SSIMC.update(SSIM)
            psnrRootC.update(psnr_Root)
        print("--->This---Our-Avg--PSNR: {:.4f} dB---ROOT-Avg--PSNR: {:.4f} dB-- SSIM:{:.4f}".format(psnrC.avg, psnrRootC.avg, SSIMC.avg))

    print("REDS4-clip_020")
    del_train_feat("test_EF")
    test_set = dataset_E.test_data_set("datasets/test/REDS4/", "020/")
    test_loader = DataLoader(dataset=test_set, batch_size=1, num_workers=1)
    psnrD = AverageMeter()
    SSIMD = AverageMeter()
    psnrRootD = AverageMeter()
    with torch.no_grad():
        model = model.cuda()
        model.eval()
        for batch in test_loader:
            ID = "test_EF"
            input, target, tID = batch
            input = input.cuda()
            target = target.cuda()
            out = model(input)
            SSIM = compute_ssim_tensor(out[0,:,:,:], target[0,:,:,:])
            psnr_our = calc_psnr(out, target)
            psnr_Root = calc_psnr(
                F.interpolate(input[:, 0, :, :, :], scale_factor=4, mode='bilinear', align_corners=False), target)
            psnrD.update(psnr_our)
            SSIMD.update(SSIM)
            psnrRootD.update(psnr_Root)
        print("--->This---Our-Avg--PSNR: {:.4f} dB---ROOT-Avg--PSNR: {:.4f} dB-- SSIM:{:.4f}".format(psnrD.avg, psnrRootD.avg, SSIMD.avg))


    print("REDS4--avg_PSNR:")
    REDS_PSNR = (psnrA.avg+psnrB.avg+psnrC.avg+psnrD.avg)/4
    print(REDS_PSNR)
    print("REDS4--avg_SSIM:")
    REDS_SSIM = (SSIMA.avg + SSIMB.avg + SSIMC.avg + SSIMD.avg) / 4
    print(REDS_SSIM)

    print("Vid4-calendar")
    del_train_feat("test_EF")
    test_set = dataset_E.test_data_set("datasets/test/test_Vid4/", "calendar/")
    test_loader = DataLoader(dataset=test_set, batch_size=1, num_workers=1)
    psnrE = AverageMeter()
    SSIME = AverageMeter()
    psnrRootE = AverageMeter()
    with torch.no_grad():
        model = model.cuda()
        model.eval()
        for batch in test_loader:
            ID = "test_EF"
            input, target, tID = batch
            input = input.cuda()
            target = target.cuda()
            out = model(input)
            SSIM = compute_ssim_tensor(out[0, :, :, :], target[0, :, :, :])
            psnr_our = calc_psnr(out, target)
            psnr_Root = calc_psnr(
                F.interpolate(input[:, 0, :, :, :], scale_factor=4, mode='bilinear', align_corners=False), target)
            psnrE.update(psnr_our)
            SSIME.update(SSIM)
            psnrRootE.update(psnr_Root)
        print("--->This---Our-Avg--PSNR: {:.4f} dB---ROOT-Avg--PSNR: {:.4f} dB-- SSIM:{:.4f}".format(psnrE.avg,
                                                                                                     psnrRootE.avg,
                                                                                                     SSIME.avg))

    print("Vid4-city")
    del_train_feat("test_EF")
    test_set = dataset_E.test_data_set("datasets/test/test_Vid4/", "city/")
    test_loader = DataLoader(dataset=test_set, batch_size=1, num_workers=1)
    psnrF = AverageMeter()
    SSIMF = AverageMeter()
    psnrRootF = AverageMeter()
    with torch.no_grad():
        model = model.cuda()
        model.eval()
        for batch in test_loader:
            ID = "test_EF"
            input, target, tID = batch
            input = input.cuda()
            target = target.cuda()
            out = model(input)
            SSIM = compute_ssim_tensor(out[0, :, :, :], target[0, :, :, :])
            psnr_our = calc_psnr(out, target)
            psnr_Root = calc_psnr(
                F.interpolate(input[:, 0, :, :, :], scale_factor=4, mode='bilinear', align_corners=False), target)
            psnrF.update(psnr_our)
            SSIMF.update(SSIM)
            psnrRootF.update(psnr_Root)
        print("--->This---Our-Avg--PSNR: {:.4f} dB---ROOT-Avg--PSNR: {:.4f} dB-- SSIM:{:.4f}".format(psnrF.avg,
                                                                                                     psnrRootF.avg,
                                                                                                     SSIMF.avg))

    print("Vid4-foliage")
    del_train_feat("test_EF")
    test_set = dataset_E.test_data_set("datasets/test/test_Vid4/", "foliage/")
    test_loader = DataLoader(dataset=test_set, batch_size=1, num_workers=1)
    psnrG = AverageMeter()
    SSIMG = AverageMeter()
    psnrRootG = AverageMeter()
    with torch.no_grad():
        model = model.cuda()
        model.eval()
        for batch in test_loader:
            ID = "test_EF"
            input, target, tID = batch
            input = input.cuda()
            target = target.cuda()
            out = model(input)
            SSIM = compute_ssim_tensor(out[0, :, :, :], target[0, :, :, :])
            psnr_our = calc_psnr(out, target)
            psnr_Root = calc_psnr(
                F.interpolate(input[:, 0, :, :, :], scale_factor=4, mode='bilinear', align_corners=False), target)
            psnrG.update(psnr_our)
            SSIMG.update(SSIM)
            psnrRootG.update(psnr_Root)
        print("--->This---Our-Avg--PSNR: {:.4f} dB---ROOT-Avg--PSNR: {:.4f} dB-- SSIM:{:.4f}".format(psnrG.avg,
                                                                                                     psnrRootG.avg,
                                                                                                     SSIMG.avg))

    print("Vid4-walk")
    del_train_feat("test_EF")
    test_set = dataset_E.test_data_set("datasets/test/test_Vid4/", "walk/")
    test_loader = DataLoader(dataset=test_set, batch_size=1, num_workers=1)
    psnrH = AverageMeter()
    SSIMH = AverageMeter()
    psnrRootH = AverageMeter()
    with torch.no_grad():
        model = model.cuda()
        model.eval()
        for batch in test_loader:
            ID = "test_EF"
            input, target, tID = batch
            input = input.cuda()
            target = target.cuda()
            out = model(input)
            SSIM = compute_ssim_tensor(out[0, :, :, :], target[0, :, :, :])
            psnr_our = calc_psnr(out, target)
            psnr_Root = calc_psnr(
                F.interpolate(input[:, 0, :, :, :], scale_factor=4, mode='bilinear', align_corners=False), target)
            psnrH.update(psnr_our)
            SSIMH.update(SSIM)
            psnrRootH.update(psnr_Root)
        print("--->This---Our-Avg--PSNR: {:.4f} dB---ROOT-Avg--PSNR: {:.4f} dB-- SSIM:{:.4f}".format(psnrH.avg,
                                                                                                     psnrRootH.avg,
                                                                                                     SSIMH.avg))

    print("Vid4--avg_PSNR:")
    VID_PSNR = (psnrE.avg + psnrF.avg + psnrG.avg + psnrH.avg) / 4
    print(VID_PSNR)
    print("Vid4--avg_SSIM:")
    VID_SSIM = (SSIME.avg + SSIMF.avg + SSIMG.avg + SSIMH.avg) / 4
    print(VID_SSIM)

    print(" ------------------- ")
    print("TOTAL:")
    print("REDS4--PSNR:{:.4f} dB -- SSIM{:.4f}".format(REDS_PSNR, REDS_SSIM))
    print("Vid4--PSNR:{:.4f} dB -- SSIM{:.4f}".format(VID_PSNR, VID_SSIM))

if __name__ == "__main__":
    test_train_set()