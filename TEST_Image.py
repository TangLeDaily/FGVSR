import torch
from dataset_E_1 import *
from torch.utils.data import DataLoader
from util import *
from rootmodel import RCAN

def main(name):
    model = RCAN.RCAN()
    checkpoint = torch.load("checkpoints/over/RCAN/model_epoch_88_psnr_26.2960.pth", map_location='cpu')
    model.load_state_dict(checkpoint.state_dict())
    model.cuda()
    model.eval()
    dataset = test_data_set("datasets/test/REDS5/", name)
    dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=0)
    psnr = AverageMeter()
    ssim = AverageMeter()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input, target = batch
            input = input.cuda()
            target = target.cuda()
            out = model(input)
            psnr_now = calc_psnr(out, target)
            ssim_now = cal_ssim_tensor(out[0, :, :, :], target[0, :, :, :])
            psnr.update(psnr_now)
            ssim.update(ssim_now)
            print("{} - PSNR :{} - SSIM :{}".format(i, psnr_now, ssim_now))
        print("{} -- PSNR: {} -- SSIM: {}".format(name, psnr.avg, ssim.avg))

if __name__ == "__main__":
    main("000/")
    main("011/")
    main("015/")
    main("020/")