import torch
from dataset_New_L2 import *
from torch.utils.data import DataLoader
from util import *
# from checkpoints.New_Over.PFNL_SP_L5FSR.PFNL_SP_5FSR  import *
from rootmodel.SPLSTM_64C_2FSR_Direct import *

def main(name):
    model = EFVSR()


    # state_dictA = torch.load('checkpoints/New_Over/PreTrain/EDVR_SP_L3/epoch_27_psnr_30.1312_lr_4e-05.pth', map_location='cpu')
    # state_dictA = state_dictA['model']
    # model.load_state_dict(state_dictA)
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
            # out = model(input)
            # out, last_h = model(input, last_h)

            psnr_now = calc_psnr(out, target)
            # save_pic(out[0, :, :, :], "data/{}/Intro/muCAN_SPSA1_2FSR/".format(name), "{}out_{:.2f}.png".format(i, psnr_now))
            ssim_now = cal_ssim_tensor(out[ 0, :, :, :], target[0, :, :, :])
            psnr.update(psnr_now, len(target))
            ssim.update(ssim_now, len(target))
            print("{} - PSNR :{} SSIM :{}".format(i, psnr_now, ssim_now))
        print("{} -- PSNR: {} -- SSIM: {}".format(name, psnr.avg, ssim.avg))
    return psnr.avg, ssim.avg
        # 0.81 0098157
if __name__ == "__main__":
    psnr1, ssim1 = main("000/")
    psnr2, ssim2 = main("011/")
    psnr3, ssim3 = main("015/")
    psnr4, ssim4 = main("020/")
    print("000 PSNR: {} ----  SSIM:{}".format(psnr1, ssim1))
    print("011 PSNR: {} ----  SSIM:{}".format(psnr2, ssim2))
    print("015 PSNR: {} ----  SSIM:{}".format(psnr3, ssim3))
    print("020 PSNR: {} ----  SSIM:{}".format(psnr4, ssim4))
    print("AVG PSNR: {} ----  SSIM:{}".format((psnr1 + psnr2 + psnr3 + psnr4)/4, (ssim1 + ssim2 + ssim3 + ssim4)/4))
