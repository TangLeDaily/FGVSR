from time import sleep

import cv2
import imageio
import torch
from torch import nn

from dataset_E import *
from torch.utils.data import DataLoader
from util import *
from checkpoints.New_Over.EDVR_2FSR import edvr_outCH
from torch.nn import functional as F

model_name = "EDVR_B"

def main(name):
    model = edvr_outCH.EDVR()
    checkpoint = torch.load("checkpoints/New_Over/EDVR_2FSR/model_epoch_187_psnr_29.4971.pth", map_location='cpu')
    model.load_state_dict(checkpoint.state_dict())
    model.cuda()
    model.eval()
    dataset = test_data_set("datasets/train/", name)
    dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=0)
    psnr = AverageMeter()
    ssim = AverageMeter()
    with torch.no_grad():
        last_h = None
        last_c = None
        for i, batch in enumerate(dataloader):
            input, target = batch
            input = input.cuda()
            target = target.cuda()
            out, out_start, after_dot, after_attn = model(input)
            psnrr = calc_psnr(out, target)
            # save_pic_cv(out_c64[0,32:33,:,:].cpu(), "data/AppendColor/{}/{}/sp".format(model_name, name),
            #            "/{}out_{:.2f}.png".format(i, psnrr))
            save_pic(out[0, :, :, :], "data/AppendColor/{}/{}/out".format(model_name, name),
                        "/{}out_{:.2f}.png".format(i, psnrr))
            # save_pic_cv2(out_h64[0, 56:57, :, :].cpu(), "data/AppendColor/{}/{}/h".format(model_name, name),
            #             "/{}out_{:.2f}.png".format(i, psnrr))
            save_pic_cv2(out_start[0, 32:33, :, :].cpu(), "data/AppendColor/{}/{}/start".format(model_name, name),
                        "/{}out_{:.2f}.png".format(i, psnrr))
            # 23
            # for j in range(64):
            #     save_pic_cv3(after_dot[0, j:j + 1, :, :].cpu(),
            #                  "data/AppendColor/{}/{}/after_dot".format(model_name, name),
            #                  "/{}out_{}.png".format(i, j))
            #
            save_pic_cv3(after_dot[0, 23:24, :, :].cpu(), "data/AppendColor/{}/{}/after_dot_t".format(model_name, name),
                         "/{}out_{:.2f}.png".format(i, psnrr))
            save_pic_cv3(after_attn[0, 0:1, :, :].cpu(), "data/AppendColor/{}/{}/after_attn".format(model_name, name),
                         "/{}out_{:.2f}.png".format(i, psnrr))
            print(i)
    return psnr.avg, ssim.avg
        # 0.8140098157
if __name__ == "__main__":
    psnr2, ssim2 = main("002/")

