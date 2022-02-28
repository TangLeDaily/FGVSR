import torch
from dataset_E import *
from torch.utils.data import DataLoader
from util import *
from checkpoints.New_Over.EDVR_2FSR import edvr_outCH

def main(name):
    model = edvr_outCH.EDVR()
    checkpoint = torch.load("checkpoints/New_Over/EDVR_2FSR/model_epoch_187_psnr_29.4971.pth", map_location='cpu')
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
            out, out_start = model(input)

            # save_pic(out[0, :, :, :], "data/Our_4-24-EDVR/{}/out".format(name), "/{}out.png".format(i))
            # save_pic(target[0, :, :, :], "data/Our_4-24-EDVR/{}/target".format(name), "/{}target.png".format(i))
            # save_pic(input[0, 0, :, :, :], "data/Our_4-24-EDVR/{}/input".format(name), "/{}input.png".format(i))
            save_pic(out_start[0, :, :, :], "data/Our_4-24-EDVR/{}/out_start".format(name), "/{}out_start.png".format(i))
            # save_pic(out_after[0, :, :, :], "data/Our_4-9/{}/out_after".format(name), "/{}out_after.png".format(i))




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
    print("000 PSNR: {} ----  SSIM:{}".format(psnr1, ssim1))
    print("011 PSNR: {} ----  SSIM:{}".format(psnr2, ssim2))
    print("015 PSNR: {} ----  SSIM:{}".format(psnr3, ssim3))
    print("020 PSNR: {} ----  SSIM:{}".format(psnr4, ssim4))
    print("AVG PSNR: {} ----  SSIM:{}".format((psnr1 + psnr2 + psnr3 + psnr4)/4, (ssim1 + ssim2 + ssim3 + ssim4)/4))
