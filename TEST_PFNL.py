import torch
from torch import optim
from torch.distributed import reduce

import wandb

# from checkpoints.over.TSALSTM_ATD.EFVSR_M_TSALSTM_DSA_AddThenDot import *
from rootmodel.PFNL import *
# from rootmodel.muCAN import Network
# from rootmodel.EFVSR_L_OnlypcdThreeDSLSTM_V3 import *
from dataset_E import *
# from rootmodel import edvr_deblur
from torch.utils.data import DataLoader
from thop import profile

def main():
    model = PFNL()
    # model = Network()
    # model = edvr_deblur.EDVR()
    model.cuda()
    input = torch.randn(1, 2, 3, 180, 320)
    input = input.cuda()
    flops, params = profile(model, inputs=(input,))
    # flops, params = profile(model, inputs=(input, None, None))
    print("model: Flops:{}  Params:{}".format(flops,params))

if __name__ == "__main__":
    main()