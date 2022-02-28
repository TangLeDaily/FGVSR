import torch
from dataset_E import *
from torch.utils.data import DataLoader
from util import *
# from checkpoints.New_Over.PFNL_SP_L5FSR.PFNL_SP_5FSR  import *
from rootmodel.SPLSTM_64C_2FSR_Direct import *

def main(name):
    model = EFVSR()
    checkpoint = torch.load("checkpoints/New_Over/EDVR_SP_2FSR/model_epoch_181_psnr_29.5809.pth", map_location='cpu')
    model.load_state_dict(checkpoint.state_dict())
    model.cuda()
    model.eval()
    dataset = test_data_set("datasets/test/", name)
    dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=0)
    psnr = AverageMeter()
    ssim = AverageMeter()
    PSNR = []
    with torch.no_grad():
        last_h = None
        last_c = None
        for i, batch in enumerate(dataloader):
            input, target = batch
            input = input.cuda()
            target = target.cuda()
            out, last_h, last_c = model(input, last_h, last_c)
            # out = model(input)
            # out, last_h = model(input, last_h)

            psnr_now = calc_psnr(out, target)
            PSNR.append(psnr_now)
            if i == 12:
                break
            # save_pic(out[0, :, :, :], "data/{}/Intro/MuCAN_SP_L5FSR/".format(name), "{}out_{:.2f}.png".format(i, psnr_now))
            ssim_now = cal_ssim_tensor(out[ 0, :, :, :], target[0, :, :, :])
            psnr.update(psnr_now, len(target))
            ssim.update(ssim_now, len(target))
            print("{} - PSNR :{} SSIM :{}".format(i, psnr_now, ssim_now))
        print("{} -- PSNR: {} -- SSIM: {}".format(name, psnr.avg, ssim.avg))
    return PSNR
        # 0.81 0098157
if __name__ == "__main__":
    psnr1 = main("000/")
    psnr2 = main("011/")
    psnr3 = main("015/")
    psnr4 = main("020/")
    for i in range(len(psnr1)):
        print("{} : {}".format(i, (psnr1[i]+psnr2[i]+psnr3[i]+psnr4[i])/4.0))

