import argparse
import os
import random
from torch.backends import cudnn
from torch.utils.data import DataLoader

from dataset import *
from util import *
from model import *

parser = argparse.ArgumentParser(description="PyTorch Data_Pre")
parser.add_argument("--train_root_path", default='datasets/train/', type=str, help="train root path")
parser.add_argument("--test_root_path", default='datasets/test/REDS4/', type=str, help="test root path")
parser.add_argument("--cuda", default=True, action="store_true", help="use cuda?")
parser.add_argument("--model_mark", default=0, type=int, help="which model to train? 0:default")
parser.add_argument("--resume", default='', type=str, help="path to latest checkpoint (default: none)")
parser.add_argument("--start_epoch", default=0, type=int, help="manual epoch number (useful on restarts)")
parser.add_argument("--batchSize", type=int, default=2, help="training batch size")  # default 16
parser.add_argument("--nEpochs", type=int, default=10000, help="number of epochs to train for")
parser.add_argument("--miniEpochs", type=int, default=2, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate. Default=1e-4")
parser.add_argument("--threads", type=int, default=8, help="number of threads for data loader to use")
parser.add_argument("--scale", type=int, default=4, help="Scale default:4x")
parser.add_argument("--loss", type=int, default=0, help="the loss function, default")
use_wandb = True
opt = parser.parse_args()
min_avr_loss = 99999999
save_flag = 0
epoch_avr_loss = 0
n_iter = 0
in_nc = 3
out_nc = 3


def main():
    global model, opt
    print(opt)

    print("===> Find Cuda")
    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    opt.seed = random.randint(1, 10000)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True

    print("===> Building model")
    if opt.model_mark == 0:
        model = SRModel(in_c=3, out_c=3, mid_c=64, kernel_size=3, pre_RN=5)

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
    print("===> Do Resume Or Skip")
    checkpoint = torch.load("checkpoints/default/model_videoID_077_epoch_0.pth")
    model.load_state_dict(checkpoint.state_dict())
    print("===> testing")
    test_total_set(model)

def del_test_feat():
    if os.path.exists("data/feature/test.npy"):
        os.remove("data/feature/test.npy")


def test_total_set(this_model):
    print(" -- Start eval --")
    del_test_feat()
    test_set = test_data_set(opt.test_root_path, "000/")
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=True, num_workers=opt.threads)
    psnr = AverageMeter()

    with torch.no_grad():
        model = this_model
        if opt.cuda:
            model = model.cuda()
        model.eval()
        for i in range(1,100):
          psnr_min = AverageMeter()
          for iteration, batch in enumerate(test_loader, 1):
              input, target, ID = batch
              if opt.cuda:
                  input = input.cuda()
                  target = target.cuda()
              out = model(input, ID)
              if i==99:
                psnr.update(calc_psnr(out, target), len(out))
              psnr_min.update(calc_psnr(out, target), len(out))
          print("--->This--i:{}--TotalAvg--PSNR: {:.4f} dB--Root--PSNR: 24.11 dB".format(i, psnr_min.avg))
        print("--->This--epoch--TotalAvg--PSNR: {:.4f} dB--Root--PSNR: 24.11 dB".format(psnr.avg))


if __name__ =="__main__":
    main()
