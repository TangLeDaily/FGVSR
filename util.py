import os

import PIL
import cv2
import imageio
import torch
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision import utils as vutils
from skimage.metrics import structural_similarity as ssim

from skimage.metrics import peak_signal_noise_ratio


def transform_convert(img_tensor):
    """
    param img_tensor: tensor
    param transforms: torchvision.transforms
    """
    transform = transforms.Compose([transforms.ToTensor()])
    if 'Normalize' in str(transform):
        normal_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform.transforms))
        mean = torch.tensor(normal_transform[0].mean, dtype=img_tensor.dtype, device=img_tensor.device)
        std = torch.tensor(normal_transform[0].std, dtype=img_tensor.dtype, device=img_tensor.device)
        img_tensor.mul_(std[:, None, None]).add_(mean[:, None, None])

    img_tensor = img_tensor.transpose(0, 2).transpose(0, 1)  # C x H x W  ---> H x W x C
    if 'ToTensor' in str(transform) or img_tensor.max() < 1:
        img_tensor = img_tensor.detach().numpy() * 255

    if isinstance(img_tensor, torch.Tensor):
        img_tensor = img_tensor.numpy()
    if img_tensor.shape[2] == 3:
        img = Image.fromarray(img_tensor.astype('uint8')).convert('RGB')
    elif img_tensor.shape[2] == 1:
        img = img_tensor.squeeze() #Image.fromarray(img_tensor.astype('uint8')).squeeze()
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_tensor.shape[2]))

    return img

def cal_psnr_tensor(a,b):
    a = a.mul(255).byte()  # 取值范围
    a = a.cpu().numpy().transpose((1, 2, 0))  # 改变数据大小
    b = b.mul(255).byte()  # 取值范围
    b = b.cpu().numpy().transpose((1, 2, 0))  # 改变数据大小
    return peak_signal_noise_ratio(a, b)

def save_pic(img_tensor, dir, name):
    # img_tensor = img_tensor.cpu()
    # img = transform_convert(img_tensor)
    if not os.path.exists(dir):
        os.makedirs(dir)
    # img.save(dir+name)
    vutils.save_image(img_tensor, dir+name, normalize=True)



def save_pic_cv(img_tensor, dir, name):
    # img_tensor = img_tensor.cpu()
    # img = transform_convert(img_tensor)
    if not os.path.exists(dir):
        os.makedirs(dir)
    # img.save(dir+name)
    # vutils.save_image(img_tensor, dir+name, normalize=True)
    img_np = transform_convert(img_tensor)
    height, width = img_np.shape



    # plt.colorbar()
    # 无边框
    fig, ax = plt.subplots()
    ax.imshow(img_np, cmap='rainbow', vmax=255.0, vmin=-255.0)  # magma_r
    fig.set_size_inches(width / 100.0 , height / 100.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.axis('off')
    plt.savefig(dir+name, dpi=300)
    plt.close('all')

def save_pic_cv2(img_tensor, dir, name):
    # img_tensor = img_tensor.cpu()
    # img = transform_convert(img_tensor)
    if not os.path.exists(dir):
        os.makedirs(dir)
    # img.save(dir+name)
    # vutils.save_image(img_tensor, dir+name, normalize=True)
    img_np = transform_convert(img_tensor)
    height, width = img_np.shape



    # plt.colorbar()
    # 无边框
    fig, ax = plt.subplots()
    ax.imshow(img_np, cmap='terrain', vmax=200.0, vmin=-200.0)  # magma_r
    fig.set_size_inches(width / 100.0 , height / 100.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.axis('off')
    plt.savefig(dir+name, dpi=300)
    plt.close('all')

import random
def rand():
    return random.random()

def save_pic_cv3(img_tensor, dir, name):
    # img_tensor = img_tensor.cpu()
    # img = transform_convert(img_tensor)
    if not os.path.exists(dir):
        os.makedirs(dir)
    # img.save(dir+name)
    # vutils.save_image(img_tensor, dir+name, normalize=True)
    img_np = transform_convert(img_tensor)
    height, width = img_np.shape



    # plt.colorbar()
    # 无边框
    fig, ax = plt.subplots()
    ax.imshow(img_np, cmap='terrain', vmax=200.0, vmin=-200.0)  # magma_r

    fig.set_size_inches(width / 100.0 , height / 100.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.axis('off')

    plt.savefig(dir+name, dpi=300)
    plt.close('all')

def save_pic_cv4(img_tensor, dir, name):
    # img_tensor = img_tensor.cpu()
    # img = transform_convert(img_tensor)
    if not os.path.exists(dir):
        os.makedirs(dir)
    # img.save(dir+name)
    # vutils.save_image(img_tensor, dir+name, normalize=True)
    img_np = transform_convert(img_tensor)
    height, width = img_np.shape
    plt.imshow(img_np, cmap='terrain', vmax=-1.0, vmin=1.0)  # magma_r
    plt.colorbar()
    # plt.subplots_adjust(top=1, bottom=1, left=0, right=1, hspace=0, wspace=0)
    plt.axis('off')
    plt.savefig(dir+name, dpi=300)
    plt.close('all')

def save_image_tensor2cv2(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为cv2格式
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    # 去掉批次维度
    input_tensor = input_tensor.squeeze(0)
    print(input_tensor.shape)
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    # RGB转BRG
    input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_GRAY2BGR)
    print(input_tensor.shape)
    cv2.imwrite(filename, input_tensor)

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

# 输入：两个tensor（可以含Batchsize维度）
def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / (torch.mean((img1 - img2) ** 2)))


import numpy as np
from PIL import Image
from scipy.signal import convolve2d
import torchvision.transforms as transforms

def tree_to_one(imgg):
    img = PIL.Image.fromarray(imgg[0,0,:,:])
    input_transform = transforms.Compose([
       transforms.Grayscale(1), #这一句就是转为单通道灰度图像
    ])
    return input_transform(img)

def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)


def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
    window = window / np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1 * im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2 * im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1 * im2, window, 'valid') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(np.mean(np.mean(ssim_map)))
def compute_ssim_tensor(tensor1, tensor2):
    a = transform_convert(tensor1.contiguous().cpu()).convert('L')
    b = transform_convert(tensor2.contiguous().cpu()).convert('L')#.contiguous().cpu()
    return compute_ssim(np.array(a), np.array(b))
def cal_ssim(a,b):
    return ssim(a, b, multichannel=True)
def cal_ssim_tensor(a,b):
    a = a.mul(255).byte()  # 取值范围
    a = a.cpu().numpy().transpose((1, 2, 0))  # 改变数据大小
    b = b.mul(255).byte()  # 取值范围
    b = b.cpu().numpy().transpose((1, 2, 0))  # 改变数据大小
    return ssim(a, b, multichannel=True)
if __name__ == "__main__":
    im1 = Image.open("datasets/denoise/test/input/000/00000000.png")
    im2 = Image.open("datasets/denoise/test/target/000/00000000.png")

    tran = transforms.ToTensor()
    im_t1 = tran(im1)
    im_t2 = tran(im2)
    #0.84900

    print(compute_ssim_tensor(im_t1, im_t2))
    im_n1 = np.asarray(im1)
    im_n2 = np.asarray(im2)
    t = cal_ssim(im_n1, im_n2)
    print(t)

    c = cal_ssim_tensor(im_t1, im_t2)
    print(c)
