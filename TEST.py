import time

import torch
from torch.backends import cudnn
import torch.nn.functional as F
import wandb

from dataset_E_TEST import *
from torch.utils.data import DataLoader
from util import *
# get model py
# import rootmodel.EFVSR_M_OnlypcdThreeLSTM
import checkpoints.over.TSALSTM_ATD.EFVSR_M_TSALSTM_DSA_AddThenDot as F1
import checkpoints.over.TSALSTM_ATD.EFVSR_M_TSALSTM_DSA_AddThenDot as F2
import rootmodel.edvr
import rootmodel.edvr_l
from rootmodel.PFNL import PFNL
# import checkpoints.over.TSALSTM_DSA_CellConv.EFVSR_M_TSALSTM_DSAdd_CellConv as F1
# import checkpoints.over.TSALSTM_DSAdd.EFVSR_M_TSALSTM_DSA_Add as F1

# use wandb
use_wandb = False
# use model
x1 = True
x2 = False
y1 = False
y2 = False
# model
model_x1 = F1.EFVSR
model_x2 = F2.EFVSR
model_y1 = PFNL
model_y2 = rootmodel.edvr_l.EDVR
# model_path
path_x1 = "checkpoints/over/TSALSTM_ATD/model_epoch_88_psnr_27.3677.pth"
path_x2 = ""
# checkpoints/over/EDVR_M_27d32/model_epoch_325_psnr_27.3234.pth
path_y1 = "checkpoints/over/PFNL_26d85/model_epoch_484_psnr_26.9643.pth"
path_y2 = ""
# dataset
pathA = ["datasets/test/REDS4/", "datasets/test/REDS4/", "datasets/test/REDS4/", "datasets/test/REDS4/"]
pathB = ["000/", "011/", "015/", "020/"]

# pathA = ["datasets/test/OurTest/"]
# pathB = ["000/"]
# pathA = ["datasets/test/REDS/", "datasets/test/REDS/", "datasets/test/REDS/", "datasets/test/REDS/"]
# pathB = ["001/", "009/", "016/", "021/"]


class Test:
    def __init__(self):
        super(Test, self).__init__()
        self.model_our = []
        self.model_other = []
        self.dataset_name = []
        self.dataloader = []

    def set_model_list(self, model1=None, model2=None, model3=None, model4=None):
        if x1:
            self.model_our.append(model1())
        if x2:
            self.model_our.append(model2())
        if y1:
            self.model_other.append(model3())
        if y2:
            self.model_other.append(model4())

    def load_model(self, model, path):
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint.state_dict())
        return model

    def load_model_list(self, path1=None, path2=None, path3=None, path4=None):
        if x1:
            self.model_our[0] = self.load_model(self.model_our[0], path1)
        if x2:
            self.model_our[1] = self.load_model(self.model_our[1], path2)
        if y1:
            self.model_other[0] = self.load_model(self.model_other[0], path3)
        if y2:
            self.model_other[1] = self.load_model(self.model_other[1], path4)

    def load_dataset_list(self, pathA, pathB):
        for i in range(len(pathA)):
            dataset = test_data_set(pathA[i], pathB[i])
            self.dataset_name.append("PATH: {} miniPATH: {} ".format(pathA[i], pathB[i]))
            self.dataloader.append(DataLoader(dataset=dataset, batch_size=1, num_workers=0))

    def cal_psnr_ssim(self, out, target):
        psnr = calc_psnr(out, target)
        ssim = compute_ssim_tensor(out[0,:,:,:], target[0,:,:,:])
        return psnr, ssim

    def start(self):
        print("===> Find Cuda")
        if not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")
        seed = random.randint(1, 10000)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        cudnn.benchmark = True

        print("===> Building model")
        self.set_model_list(model_x1, model_x2, model_y1, model_y2)
        self.load_model_list(path_x1, path_x2, path_y1, path_y2)

        print("===> Building dataset")
        self.load_dataset_list(pathA, pathB)
        TimeX1 = 0
        TimeX2 = 0
        TimeY1 = 0
        TimeY2 = 0


        print("===> Start test")
        with torch.no_grad():
            for i, daloader in enumerate(self.dataloader):
                last_h1 = None
                last_h2 = None
                last_c1 = None
                last_c2 = None
                print("===> Set Var of PSNR and SSIM and CUDA and EVAL")
                PSNR_OUR = []
                SSIM_OUR = []
                for j in range(len(self.model_our)):
                    PSNR_OUR.append(AverageMeter())
                    SSIM_OUR.append(AverageMeter())
                    self.model_our[j] = self.model_our[j].cuda()
                    self.model_our[j].eval()
                PSNR_ROOT = AverageMeter()
                SSIM_ROOT = AverageMeter()
                PSNR_OTHER = []
                SSIM_OTHER = []
                PSNR_OUR_WEN = [0, 0]
                PSNR_OTHER_WEN = [0, 0]
                last_psnr_our_1 = 0
                last_psnr_our_2 = 0
                last_psnr_other_1 = 0
                last_psnr_other_2 = 0
                for j in range(len(self.model_other)):
                    PSNR_OTHER.append(AverageMeter())
                    SSIM_OTHER.append(AverageMeter())
                    self.model_other[j] = self.model_other[j].cuda()
                    self.model_other[j].eval()
                if use_wandb:
                    wandb.init(project="TEST_NEW", name=str(i), entity="karledom")
                print("===> this dataset is {}".format(self.dataset_name[i]))
                for iter, batch in enumerate(daloader):
                    input, target = batch
                    input = input.cuda()
                    target = target.cuda()
                    output_our_psnr = []
                    output_our_ssim = []
                    output_other_psnr = []
                    output_other_ssim = []
                    base_output = F.interpolate(input[:,0,:,:,:], scale_factor=4, mode='bilinear', align_corners=False)
                    base_psnr, base_ssim = self.cal_psnr_ssim(base_output, target)
                    PSNR_ROOT.update(base_psnr)
                    SSIM_ROOT.update(base_ssim)
                    if use_wandb:
                        wandb.log({"base_psnr": base_psnr, "base_ssim": base_ssim, "iter": iter})
                    if x1:
                        timestart = time.time()
                        out, last_h1, last_c1 = self.model_our[0](input, last_h=last_h1, last_c=last_c1)
                        timeover = time.time()
                        TimeX1 += (timeover-timestart)
                        psnr, ssim = self.cal_psnr_ssim(out, target)
                        PSNR_OUR[0].update(psnr)
                        SSIM_OUR[0].update(ssim)
                        output_our_psnr.append(psnr)
                        output_our_ssim.append(ssim)
                        del_psnr = psnr - base_psnr
                        del_ssim = ssim - base_ssim
                        if iter == 0:
                            last_psnr_our_1 = psnr
                        PSNR_OUR_WEN[0] = PSNR_OUR_WEN[0] + abs(psnr-last_psnr_our_1)
                        if use_wandb:
                            wandb.log({"out_M_psnr": psnr, "our_M_ssim": ssim, "iter": iter})
                            wandb.log({"M_base_del_psnr": del_psnr, "M_base_del_ssim": del_ssim, "iter": iter})
                    if x2:
                        timestart = time.time()
                        out, last_h2, last_c2 = self.model_our[1](input, last_h=last_h2, last_c=last_c2)
                        timeover = time.time()
                        TimeX2 += (timeover - timestart)
                        psnr, ssim = self.cal_psnr_ssim(out, target)
                        PSNR_OUR[1].update(psnr)
                        SSIM_OUR[1].update(ssim)
                        output_our_psnr.append(psnr)
                        output_our_ssim.append(ssim)
                        del_psnr = psnr - base_psnr
                        del_ssim = ssim - base_ssim
                        if iter == 0:
                            last_psnr_our_2 = psnr
                        PSNR_OUR_WEN[1] = PSNR_OUR_WEN[1] + abs(psnr-last_psnr_our_2)
                        if use_wandb:
                            wandb.log({"out_L_psnr": psnr, "our_L_ssim": ssim, "iter": iter})
                            wandb.log({"L_base_del_psnr": del_psnr, "L_base_del_ssim": del_ssim, "iter": iter})
                    if y1:
                        timestart = time.time()
                        out = self.model_other[0](input)
                        timeover = time.time()
                        TimeY1 += (timeover - timestart)

                        psnr, ssim = self.cal_psnr_ssim(out, target)
                        PSNR_OTHER[0].update(psnr)
                        SSIM_OTHER[0].update(ssim)
                        output_other_psnr.append(psnr)
                        output_other_ssim.append(ssim)
                        del_psnr = psnr - base_psnr
                        del_ssim = ssim - base_ssim
                        if iter == 0:
                            last_psnr_other_1 = psnr
                        PSNR_OTHER_WEN[0] = PSNR_OTHER_WEN[0] + abs(psnr-last_psnr_other_1)
                        if use_wandb:
                            wandb.log({"other_M_psnr": psnr, "other_M_ssim": ssim, "iter": iter})
                            wandb.log({"otherM_base_del_psnr": del_psnr, "otherM_base_del_ssim": del_ssim, "iter": iter})
                    if y2:
                        timestart = time.time()
                        out = self.model_other[1](input)
                        timeover = time.time()
                        TimeY2 += (timeover - timestart)
                        psnr, ssim = self.cal_psnr_ssim(out, target)
                        PSNR_OTHER[1].update(psnr)
                        SSIM_OTHER[1].update(ssim)
                        output_other_psnr.append(psnr)
                        output_other_ssim.append(ssim)
                        del_psnr = psnr - base_psnr
                        del_ssim = ssim - base_ssim
                        if iter == 0:
                            last_psnr_other_2 = psnr
                        PSNR_OTHER_WEN[1] = PSNR_OTHER_WEN[1] + abs(psnr-last_psnr_other_2)
                        if use_wandb:
                            wandb.log({"other_L_psnr": psnr, "other_L_ssim": ssim, "iter": iter})
                            wandb.log({"otherL_base_del_psnr": del_psnr, "otherL_base_del_ssim": del_ssim, "iter": iter})

                    if x1 and y1:
                        del_psnr = output_our_psnr[0]-output_other_psnr[0]
                        del_ssim = output_our_ssim[0]-output_other_ssim[0]
                        if use_wandb:
                            wandb.log({"M_otherM_del_psnr": del_psnr, "M_otherM_del_ssim": del_ssim, "iter": iter})
                    if x2 and y2:
                        del_psnr = output_our_psnr[1]-output_other_psnr[1]
                        del_ssim = output_our_ssim[1]-output_other_ssim[1]
                        if use_wandb:
                            wandb.log({"L_ptherL_del_psnr": del_psnr, "L_therL_del_ssim": del_ssim, "iter": iter})
                    if x1 and x2:
                        del_psnr = output_our_psnr[1]-output_our_psnr[0]
                        del_ssim = output_our_ssim[1]-output_our_ssim[0]
                        if use_wandb:
                            wandb.log({"L_M_our_del_psnr": del_psnr, "L_M_our_del_ssim": del_ssim, "iter": iter})
                    if y1 and y2:
                        del_psnr = output_other_psnr[1]-output_other_psnr[0]
                        del_ssim = output_other_ssim[1]-output_other_ssim[0]
                        if use_wandb:
                            wandb.log({"L_M_other_del_psnr": del_psnr, "L_M_other_del_ssim": del_ssim, "iter": iter})
                print("Time: X1:{}, X2:{}, Y1:{}, Y2:{}".format(TimeX1, TimeX2, TimeY1, TimeY2))
                for k in range(len(self.model_our)):
                    print("our_model:{}--PSNR:{:.4f}--SSIM:{:.4f}".format(k, PSNR_OUR[k].avg, SSIM_OUR[k].avg))
                for k in range(len(self.model_other)):
                    print("other_model:{}--PSNR:{:.4f}--SSIM:{:.4f}".format(k, PSNR_OTHER[k].avg, SSIM_OTHER[k].avg))


                print("base--PSNR:{:.4f}--SSIM:{:.4f}".format(PSNR_ROOT.avg, SSIM_ROOT.avg))




                if use_wandb:
                    wandb.finish()




def main():
    Test().start()


if __name__ == "__main__":
    main()
