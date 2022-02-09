import os
import random
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
from torch.utils.data import DataLoader

def load_img(filepath):
    img = Image.open(filepath)
    return img

def get_pic_name_lis(rootpath, path): #datasets/train/   and   000/
    imgls = []
    for i in os.listdir(rootpath + "input/" + path):
        imgls.append(i)
    imglss = sorted(imgls)
    return imglss

def get_path(rootpath): #datasets/train/
    pathls = []
    for i in os.listdir(rootpath + "input/"):
        pathls.append(i+"/")
    return pathls

def get_total_path(rootpath, mid, pathls):
    out = []
    for path in pathls:
        out.append(rootpath + mid + path)
    return out



class train_data_set(data.Dataset):
    def __init__(self, rootpath, batchsize): #datasets/train/
        super(train_data_set, self).__init__()
        self.batchsize = batchsize
        self.pathls = get_path(rootpath) # 000/ 001/ 002/
        self.input_path_ls = get_total_path(rootpath, "input/", self.pathls)  #datasets/train/input/000/
        self.target_path_ls = get_total_path(rootpath, "target/", self.pathls)
        self.pic_name_lis = get_pic_name_lis(rootpath, self.pathls[0])
        self.transLR = transforms.Compose([
            transforms.ToTensor()
             ,transforms.CenterCrop(128)
            ])
        self.transHR = transforms.Compose([
            transforms.ToTensor()
             ,transforms.CenterCrop(128*4)
            ])

    def __len__(self):
        return len(self.input_path_ls) * len(self.pic_name_lis)

    def __getitem__(self, idx):
        video_Bo = idx // (self.batchsize * len(self.pic_name_lis)) # 16 // 8 * 100 = 0,  821 // 8 * 100 = 1
        video_ID = idx % self.batchsize # 16 % 8 = 0, 821 % 8 = 5
        video_real_ID = self.batchsize * video_Bo + video_ID # 0*8 + 0 = 0,  1*8 + 5 = 13
        frame = idx % (self.batchsize * len(self.pic_name_lis)) // self.batchsize # 16 % 800 // 8 = 2, 821 % 800 // 8 = 21 // 8 = 2
        feat_ID = video_real_ID
        lr = self.transLR(load_img(self.input_path_ls[video_real_ID] + self.pic_name_lis[frame]))
        hr = self.transHR(load_img(self.target_path_ls[video_real_ID] + self.pic_name_lis[frame]))
        return lr, hr, feat_ID
class test_data_set(data.Dataset):
    def __init__(self, rootpath, path): #datasets/test/  and 000/
        super(test_data_set, self).__init__()
        self.feat_ID = "test"
        self.input_path = rootpath + "input/" + path  #datasets/test/input/000/
        self.target_path = rootpath + "target/" + path
        self.pic_name_lis = get_pic_name_lis(rootpath, path)
        self.transLR = transforms.Compose([
            transforms.ToTensor()])
        self.transHR = transforms.Compose([
            transforms.ToTensor()])

    def __len__(self):
        return len(self.pic_name_lis)

    def __getitem__(self, idx):
        lr = self.transLR(load_img(self.input_path + self.pic_name_lis[idx]))
        hr = self.transHR(load_img(self.target_path + self.pic_name_lis[idx]))
        return lr, hr,  self.feat_ID
