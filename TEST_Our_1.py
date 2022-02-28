import torch
from dataset_E_1 import *
from torch.utils.data import DataLoader
from util import *
from checkpoints.over.RCAN_our import RCAN_our

def main(name):
    model = RCAN_our.RCAN()
    checkpoint = torch.load("checkpoints/over/RCAN_our/model_epoch_195_psnr_26.4425.pth", map_location='cpu')
    model.load_state_dict(checkpoint.state_dict())
    model.cuda()
    model.eval()
    dataset = test_data_set("datasets/test/REDS4/", name)
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
            # save_pic(out[0,:,:,:], "data/Our_4-1/", "{}out.png".format(i))
            psnr_now = calc_psnr(out, target)
            ssim_now = cal_ssim_tensor(out[ 0, :, :, :], target[0, :, :, :])
            psnr.update(psnr_now, len(target))
            ssim.update(ssim_now, len(target))
            print("{} - PSNR :{} SSIM :{}".format(i, psnr_now, ssim_now))
        print("{} -- PSNR: {} -- SSIM: {}".format(name, psnr.avg, ssim.avg))
    return psnr.avg, ssim.avg
        # 0.8140098157
if __name__ == "__main__":
    psnr1, ssim1 = main("000/")
    psnr2, ssim2 = main("011/")
    psnr3, ssim3 = main("015/")
    psnr4, ssim4 = main("020/")
    print("------------------")
    print("000 PSNR: {} ----  SSIM:{}".format(psnr1, ssim1))
    print("011 PSNR: {} ----  SSIM:{}".format(psnr2, ssim2))
    print("015 PSNR: {} ----  SSIM:{}".format(psnr3, ssim3))
    print("020 PSNR: {} ----  SSIM:{}".format(psnr4, ssim4))
    print("AVG PSNR: {} ----  SSIM:{}".format((psnr1 + psnr2 + psnr3 + psnr4)/4, (ssim1 + ssim2 + ssim3 + ssim4)/4))
