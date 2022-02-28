import torch
import wandb

from dataset_New_L2 import *
from torch.utils.data import DataLoader
from util import *
from rootmodel import SPLSTM_64C_2FSR_Direct
from rootmodel import edvr

wandb.init(project="TEST_DB_SUP", name="SPL_w/_forget", entity="karledom")

dpsnr_avg_list = []
dssim_avg_list = []
def main(name):
    model_our = SPLSTM_64C_2FSR_Direct.EFVSR()
    model_other = edvr.EDVR()

    # checkpoint_A = torch.load("checkpoints/New_Over/RCAN_SP_1FSR/model_epoch_70_psnr_28.8046.pth", map_location='cpu')

    # state_dictA = torch.load('checkpoints/New_Over/PreTrain/EDVR_SP_L5/epoch_17_psnr_30.2587_lr_5e-06.pth', map_location='cpu')
    # state_dictA = state_dictA['model']

    state_dictA = torch.load("checkpoints/New_Over/EDVR_SP_2FSR/model_epoch_181_psnr_29.5809.pth", map_location='cpu')
    model_our.load_state_dict(state_dictA.state_dict())
    model_our.cuda()
    model_our.eval()
    # checkpoint_B = torch.load("checkpoints/New_Over/RCAN_1FSR/model_epoch_78_psnr_28.7558.pth", map_location='cpu')

    # state_dictB = torch.load('checkpoints/New_Over/PreTrain/EDVR_L5/epoch_204_psnr_30.2176_lr_1.8889465931478612e-07.pth', map_location='cpu')
    # state_dictB = state_dictB['model']

    state_dictB = torch.load("checkpoints/New_Over/EDVR_2FSR/model_epoch_187_psnr_29.4971.pth", map_location='cpu')
    model_other.load_state_dict(state_dictB.state_dict())
    model_other.cuda()
    model_other.eval()

    dataset = test_data_set("datasets/test/", name)
    dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=0)
    psnr_A = AverageMeter()
    ssim_A = AverageMeter()
    psnr_B = AverageMeter()
    ssim_B = AverageMeter()
    dpsnr_now_list = []
    dssim_now_list = []
    with torch.no_grad():
        last_h = None
        last_c = None
        for i, batch in enumerate(dataloader):
            input, target= batch
            input = input.cuda()
            target = target.cuda()
            out_A, last_h, last_c = model_our(input, last_h, last_c)
            out_B = model_other(input)

            # cal psnr ssim
            psnr_now_A = calc_psnr(out_A, target)
            ssim_now_A = cal_ssim_tensor(out_A[0, :, :, :], target[0, :, :, :])
            psnr_A.update(psnr_now_A, len(target))
            ssim_A.update(ssim_now_A, len(target))

            psnr_now_B = calc_psnr(out_B, target)
            ssim_now_B = cal_ssim_tensor(out_B[0, :, :, :], target[0, :, :, :])
            psnr_B.update(psnr_now_B, len(target))
            ssim_B.update(ssim_now_B, len(target))

            DT_psnr = psnr_now_A - psnr_now_B
            DT_ssim = ssim_now_A - ssim_now_B
            dssim_now_list.append(DT_ssim)
            dpsnr_now_list.append(DT_psnr)
            # save_pic(out_A[0,:,:,:], "data/5_20_test/{}/EDVR_SP_L5/".format(name), "pic_{}_psnr_{:.4f}_DT_{:.4f}.png".format(i, psnr_now_A, DT_psnr))
            # save_pic(out_B[0, :, :, :], "data/5_20_test/{}/EDVR_L5/".format(name),
            #         "pic_{}_psnr_{:.4f}_DT_{:.4f}.png".format(i,psnr_now_B, DT_psnr))
            print("name_{}_pic_{}_DP_{:.4f}".format(name, i, DT_psnr))
            wandb.log({'i':i,'psnr_{}_SP'.format(name): psnr_now_A, 'psnr_{}_Base'.format(name): psnr_now_B, 'D_PSNR_{}'.format(name): psnr_now_A-psnr_now_B})
        print("{} -A- PSNR: {} -- SSIM: {}".format(name, psnr_A.avg, ssim_A.avg))
        print("{} -B- PSNR: {} -- SSIM: {}".format(name, psnr_B.avg, ssim_B.avg))
    dpsnr_avg_list.append(dpsnr_now_list)
    dssim_avg_list.append(dpsnr_now_list)
    return psnr_A.avg, ssim_A.avg
        # 0.8140098157
if __name__ == "__main__":
    psnr1, ssim1 = main("000/")
    psnr2, ssim2 = main("011/")
    psnr3, ssim3 = main("015/")
    psnr4, ssim4 = main("020/")
    psnr_avg = []
    ssim_avg = []
    for i in range(100):
        this_psnr_sum = 0
        this_ssim_sum = 0
        for j in range(4):
            this_psnr_sum = this_psnr_sum + dpsnr_avg_list[j][i]
            this_ssim_sum = this_ssim_sum + dssim_avg_list[j][i]
        this_psnr_avg = this_psnr_sum/4.
        this_ssim_avg = this_ssim_sum/4.
        wandb.log({'Frame': i, 'D_PSNR': this_psnr_avg, 'D_SSIM': this_ssim_avg})
        psnr_avg.append(this_psnr_avg)
        ssim_avg.append(this_ssim_avg)

    # print("000 PSNR: {} ----  SSIM:{}".format(psnr1, ssim1))
    # print("011 PSNR: {} ----  SSIM:{}".format(psnr2, ssim2))
    # print("015 PSNR: {} ----  SSIM:{}".format(psnr3, ssim3))
    # print("020 PSNR: {} ----  SSIM:{}".format(psnr4, ssim4))
    # print("AVG PSNR: {} ----  SSIM:{}".format((psnr1 + psnr2 + psnr3 + psnr4)/4, (ssim1 + ssim2 + ssim3 + ssim4)/4))
