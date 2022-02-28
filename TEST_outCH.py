from time import sleep

import cv2
import imageio
import torch
from torch import nn

from dataset_E import *
from torch.utils.data import DataLoader
from util import *
# from checkpoints.New_Over.RCAN_SP_1FSR.RCAN_SP_1FSR import *
from checkpoints.New_Over.PFNL_SP_2FSR.PFNL_SP_2FSR import *
from torch.nn import functional as F

model_name = "PFNL"

def main(name):
    model = PFNL()
    checkpoint = torch.load("checkpoints/New_Over/PFNL_SP_2FSR/model_epoch_162_psnr_28.9309.pth", map_location='cpu')
    model.load_state_dict(checkpoint.state_dict())
    model.cuda()
    model.eval()
    dataset = test_data_set("datasets/test/", name)
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
            # out, last_h, last_c, out_s= model(input, last_h, last_c)
            # out = bic()
            psnrr = calc_psnr(out, target)
            # print(out_s[0].size())
            # for j in range(64):
            #     save_pic_cv(out_s[0][0, j:j+1, :, :].cpu(), "data/SP/{}/{}/sp".format(model_name, name),
            #                 "/{}out_{:.2f}.png".format(j, psnrr))
            save_pic_cv(out_s[0][0,9:10,:,:].cpu(), "data/SP/{}/{}/sp".format(model_name, name),
                        "/{}out_{:.2f}.png".format(i, psnrr))
            # save_pic(out[0, :, :, :], "data/AppendColor/{}/{}/out".format(model_name, name),
            #             "/{}out_{:.2f}.png".format(i, psnrr))
            # save_pic_cv2(out_h64[0, 56:57, :, :].cpu(), "data/AppendColor/{}/{}/h".format(model_name, name),
            #             "/{}out_{:.2f}.png".format(i, psnrr))
            # save_pic_cv2(out_start64[0, 32:33, :, :].cpu(), "data/AppendColor/{}/{}/start".format(model_name, name),
            #             "/{}out_{:.2f}.png".format(i, psnrr))
            # # 18
            # for j in range(64):
            #     save_pic_cv3(after_dot[0, j:j+1, :, :].cpu(), "data/AppendColor/{}/{}/after_dot".format(model_name, name),
            #              "/{}out_{}.png".format(i, j))

            # save_pic_cv3(after_dot[0, 18:19, :, :].cpu(), "data/AppendColor/{}/{}/after_dot_t".format(model_name, name),
            #              "/{}out_{:.2f}.png".format(i, psnrr))
            # save_pic_cv4(after_attn[0, 0:1, :, :].cpu(), "data/Test_5-8/{}/{}/after_attn".format(model_name, name),
            #              "/{}out_{:.2f}.png".format(i, psnrr))
            print(i)
    return psnr.avg, ssim.avg
        # 0.8140098157
if __name__ == "__main__":
    psnr2, ssim2 = main("011/")

