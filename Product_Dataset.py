from os import listdir
from os.path import join
from torchvision.transforms import Compose,  CenterCrop, Resize
from PIL import Image
import os
from torch.nn import functional as F

def is_imagefile(image):
    return any(image.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG',
                                                               '.JPEG','bmp','BMP'])

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def hr_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
    ])

def lr_transform(crop_size):
    return Compose([
        Resize(crop_size, interpolation=Image.BICUBIC),
    ])


def produce_image(data_dir, output_lr_dir, output_hr_dir, scale):
    filename = [join(data_dir, x) for x in listdir(data_dir) if is_imagefile(x)]

    for x in filename:
        images_name = x.split('/')[-1]
        images_name = images_name.split('.')[0]
        x_image = Image.open(x)
        (w, h) = x_image.size
        print(w, h)
        nw = calculate_valid_crop_size(w, 4)
        nh = calculate_valid_crop_size(h, 4)
        hr_size = hr_transform((nh, nw))
        x_image = hr_size(x_image)
        print(images_name)
        save_image(x_image, scale, images_name, output_lr_dir, output_hr_dir)

def save_image(x_image, scale, images_name, output_lr_dir, output_hr_dir):
    if not os.path.exists(output_hr_dir):
        os.makedirs(output_hr_dir)
    if not os.path.exists(output_lr_dir):
        os.makedirs(output_lr_dir)

    x_image.save(os.path.join(output_hr_dir, images_name + '.png'), quality=95,)
    s = scale
    path = os.path.join(output_lr_dir, images_name + '.png')
    print(path)
    (nw, nh) = x_image.size
    lr_size = lr_transform((nh // s, nw // s))
    xr_image = lr_size(x_image)
    up_size = lr_transform((nh, nw))
    xr_image = up_size(xr_image)
    xr_image.save(path)

def main():
    scale = 16 # 模糊尺寸
    data_dir = "datasets/test/REDS4/target/000/" # 源图片文件夹
    data_HR = "datasets/test/OurTest/target/011" # 随便设，应该用不到
    data_LR = "datasets/test/OurTest/input/011" #  目的文件夹

    produce_image(data_dir, data_LR, data_HR, scale)


if __name__ == "__main__":
    main()
