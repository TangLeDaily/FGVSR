import torch
from torch import optim
from torch.distributed import reduce

import wandb

# from rootmodel. import *
from rootmodel.muCAN_SPSA1_5FSR import *
# from checkpoints.New_Over.EDVR_1FSR.EDVR_1FSR import *
# from checkpoints.New_Over.EDVR_GRU_2FSR.EDVR_GRU_2FSR import *
from torch.utils.data import DataLoader
from thop import profile

def main():
    # model = PFNL()
    # model = Network()
    # model = edvr_deblur_our.EDVR()
    model = Network()
    model.cuda()
    # input = torch.randn(1, 2, 3, 180, 320)
    input = torch.randn(1, 5, 3, 180, 320)
    input = input.cuda()

    with torch.no_grad():
        flops, params = profile(model, inputs=(input,None,None))
        # flops, params = profile(model, inputs=(input, None))
        # flops, params = profile(model, inputs=(input, ))
    print("model: Flops:{}  Params:{}".format(flops,params))

if __name__ == "__main__":
    main()

