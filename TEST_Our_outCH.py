from time import sleep

import cv2
import imageio
import torch
from torch import nn

from dataset_E import *
from torch.utils.data import DataLoader
from util import *
from checkpoints.New_Over.EDVR_SP_2FSR import SPLSTM_64C_2FSR_Direct_OutHC
from torch.nn import functional as F

def main(name):
    model = SPLSTM_64C_2FSR_Direct_OutHC.EFVSR()
    checkpoint = torch.load("checkpoints/New_Over/EDVR_SP_2FSR/model_epoch_181_psnr_29.5809.pth", map_location='cpu')
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
            out, last_h, last_c = model(input, last_h, last_c)

            # out_c_soft = F.softmax(out_c[0, :, :, :]*0.5, dim=2)
            BN = nn.BatchNorm2d(3, affine=True)
            BN = BN.cuda()
            # out_c_soft = BN(out_c)




            # out_c64_np = F.relu(out_c64).cpu()
            save_pic_cv(last_c[0,32:33,:,:].cpu(), "data/5_19test/out_c_orin/{}".format(name),
                        "/{}out_c_soft.png".format(i))
            # for j in range(64):
            # save_pic_cv2(out_h64[0, 56:57, :, :].cpu(), "data/Our_TestColor/out_hh_soft/{}".format(name),
                       #  "/{}out_h_soft.png".format(i))
            # save_pic_cv2(out_start64[0, 32:33, :, :].cpu(), "data/Our_TestColor/out_start_soft/{}".format(name),
            #             "/{}out_c_soft.png".format(i))

            # plt.imshow(out_c64_np[0, 0, :, :], cmap='viridis')
            # plt.savefig(facecolor="")
            # sleep(10000)
            # save_pic(out[0, :, :, :], "data/Our_4-24-sig/{}/out".format(name), "/{}out.png".format(i))
            # save_pic(target[0, :, :, :], "data/Our_4-24-sig/{}/target".format(name), "/{}target.png".format(i))
            # save_pic(input[0, 0, :, :, :], "data/Our_4-24-sig/{}/input".format(name), "/{}input.png".format(i))
            # save_pic(out_start[0, :, :, :], "data/Our_TestColor/out_start/{}".format(name), "/{}out_start.png".format(i))
            # save_pic(out_h[0, :, :, :], "data/Our_TestColor/out_h/{}".format(name), "/{}out_h.png".format(i))
            # save_pic(out_c[0, :, :, :], "data/Our_TestColor/out_c/{}".format(name), "/{}out_c.png".format(i))
            # save_pic(out_after[0, :, :, :], "data/Our_4-9/{}/out_after".format(name), "/{}out_after.png".format(i))




            # psnr_now = calc_psnr(out, target)
            # ssim_now = cal_ssim_tensor(out[ 0, :, :, :], target[0, :, :, :])
            # psnr.update(psnr_now, len(target))
            # ssim.update(ssim_now, len(target))
            # print("{} - PSNR :{} SSIM :{}".format(i, psnr_now, ssim_now))
            print(i)
        # print("{} -- PSNR: {} -- SSIM: {}".format(name, psnr.avg, ssim.avg))
    return psnr.avg, ssim.avg
        # 0.8140098157
if __name__ == "__main__":
    # psnr1, ssim1 = main("000/")
    # psnr2, ssim2 = main("011/")
    psnr3, ssim3 = main("015/")
    psnr4, ssim4 = main("020/")
    # print("000 PSNR: {} ----  SSIM:{}".format(psnr1, ssim1))
    # print("011 PSNR: {} ----  SSIM:{}".format(psnr2, ssim2))
    print("015 PSNR: {} ----  SSIM:{}".format(psnr3, ssim3))
    print("020 PSNR: {} ----  SSIM:{}".format(psnr4, ssim4))
    # print("AVG PSNR: {} ----  SSIM:{}".format((psnr1 + psnr2 + psnr3 + psnr4)/4, (ssim1 + ssim2 + ssim3 + ssim4)/4))
