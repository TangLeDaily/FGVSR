import torch
from rootmodel.EFVSR_L_OnlypcdThreeDSLSTM_V3 import *

def main():
    model = EFVSR()
    # for n, p in model.named_parameters():
    #     if "pcd_align.DSLSTM" in n:
    #         p.requires_grad = True
    #     else:
    #         p.requires_grad = False
    #     print(n, p.requires_grad)

    gates = torch.randn([8, 8, 32, 32])
    h_crop = gates.chunk(4, 2)
    w_crop = []
    h_cropp = []
    for i in range(4):
        w_crop.append(h_crop[i].chunk(4, 3))
        w_cropp = []
        for j in range(4):
            w_cropp.append(w_crop[i][j])
        h_cropp.append(torch.cat(w_cropp, 3))
    gates = torch.cat(h_cropp, 2)
    print(gates.size())


if __name__ == "__main__":
    main()