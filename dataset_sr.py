import os

import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms

def load_img(filepath):
    img = Image.open(filepath)
    return img

def get_pic_name_lis(rootpath, path): #datasets/train/   and   000/
    imgls = []
    for i in os.listdir(rootpath + "input/" + path):
        imgls.append(i)
    return imgls

def get_path(rootpath): #datasets/train/
    pathls = []
    for i in os.listdir(rootpath + "input/"):
        pathls.append(i+"/")
    return pathls
class RandomCrop(object):
    def __init__(self):
        self.scale = 4
        self.output_size = (64, 64)
    def __call__(self, sample):
        lra, lrb, hrp = sample['lra'], sample['lrb'], sample['hr']
        h, w = lra.shape[1: 3]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        new_lra = lra[:, top:top + new_h, left: left + new_w]
        new_lrb = lrb[:, top:top + new_h, left: left + new_w]
        new_hr = hrp[:, top * self.scale:top * self.scale + new_h * self.scale,
                 left * self.scale: left * self.scale + new_w * self.scale]

        return new_lra, new_lrb, new_hr

class train_data_set(data.Dataset):
    def __init__(self, rootpath, path): #datasets/train/  and 000/
        super(train_data_set, self).__init__()
        self.input_path = rootpath + "input/" + path  #datasets/train/input/000/
        self.target_path = rootpath + "target/" + path
        self.pic_name_lis = get_pic_name_lis(rootpath, path)
        self.transLR = transforms.Compose([
            transforms.ToTensor()
             # ,transforms.CenterCrop(128)
            ])
        self.transHR = transforms.Compose([
            transforms.ToTensor()
             # ,transforms.CenterCrop(128*4)
            ])
        self.crop = transforms.Compose([RandomCrop()])

    def __len__(self):
        return len(self.pic_name_lis)

    def __getitem__(self, idx):

        if idx == 0:
            refLR = self.transLR(load_img(self.input_path + self.pic_name_lis[idx]))
            nebLR = self.transLR(load_img(self.input_path + self.pic_name_lis[idx + 1]))
        else:
            refLR = self.transLR(load_img(self.input_path + self.pic_name_lis[idx]))
            nebLR = self.transLR(load_img(self.input_path + self.pic_name_lis[idx - 1]))

        hrp = self.transHR(load_img(self.target_path + self.pic_name_lis[idx]))
        sample = {'lra': refLR, 'lrb': nebLR, 'hr': hrp}
        lra, lrb, hr = self.crop(sample)
        lr = torch.stack((lra, lrb), dim=0)
        return lr, hr

class test_data_set(data.Dataset):
    def __init__(self, rootpath, path): #datasets/train/  and 000/
        super(test_data_set, self).__init__()
        self.input_path = rootpath + "input/" + path  #datasets/train/input/000/
        self.target_path = rootpath + "target/" + path
        self.pic_name_lis = get_pic_name_lis(rootpath, path)
        self.transLR = transforms.Compose([
            transforms.ToTensor()
             # ,transforms.CenterCrop(128)
            ])
        self.transHR = transforms.Compose([
            transforms.ToTensor()
             # ,transforms.CenterCrop(128*4)
            ])

    def __len__(self):
        return len(self.pic_name_lis)

    def __getitem__(self, idx):
        if idx == 0:
            refLR = self.transLR(load_img(self.input_path + self.pic_name_lis[idx]))
            nebLR = self.transLR(load_img(self.input_path + self.pic_name_lis[idx + 1]))
        else:
            refLR = self.transLR(load_img(self.input_path + self.pic_name_lis[idx]))
            nebLR = self.transLR(load_img(self.input_path + self.pic_name_lis[idx - 1]))
        lr = torch.stack((refLR, nebLR), dim=0)
        hr = self.transHR(load_img(self.target_path + self.pic_name_lis[idx]))
        return lr, hr

from torch.utils.data import DataLoader
from util import *
if __name__ == "__main__":
    data = train_data_set("datasets/train/", "000/")
    train_loader = DataLoader(dataset=data, batch_size=4, shuffle=True, num_workers=0,
                              drop_last=True)
    for iteration, batch in enumerate(train_loader, 1):
        input, target = batch
        print("input & target:")
        print(input.size())
        print(target.size())
        save_pic(target[0,:,:,:], "test/", "hr.png")
        save_pic(input[0,0,:,:,:], "test/", "lr.png")
