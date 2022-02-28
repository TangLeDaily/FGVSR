import torch
from skimage.metrics import structural_similarity as ssim
from torchvision import utils as vutils
from dataset_E_1 import *
from torch.utils.data import DataLoader
from rootmodel import RCAN


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / (torch.mean((img1 - img2) ** 2)))
def cal_ssim_tensor(a,b):
    a = a.mul(255).byte()  # 取值范围
    a = a.cpu().numpy().transpose((1, 2, 0))  # 改变数据大小
    b = b.mul(255).byte()  # 取值范围
    b = b.cpu().numpy().transpose((1, 2, 0))  # 改变数据大小
    return ssim(a, b, multichannel=True)
def save_pic(img_tensor, dir, name):
    # img_tensor = img_tensor.cpu()
    # img = transform_convert(img_tensor)
    if not os.path.exists(dir):
        os.makedirs(dir)
    # img.save(dir+name)
    vutils.save_image(img_tensor, dir+name, normalize=True)

def main(name):
    model = RCAN.RCAN()
    checkpoint = torch.load("checkpoints/over/RCAN/model_epoch_88_psnr_26.2960.pth", map_location='cpu')
    model.load_state_dict(checkpoint)
    # model.load_state_dict(checkpoint.state_dict())
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
            save_pic(out[0, :, :, :], "data/Our_4-14/{}/out".format(name), "/{}out.png".format(i))
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