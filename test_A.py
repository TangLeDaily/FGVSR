# coding=gbk

import os

# 文件复制
# src_path = r'C:\Users\RSB\Desktop\Python文件夹\p1'
# target_path = r'C:\Users\RSB\Desktop\Python文件夹\p3'
#
# filelist = os.listdir(src_path)
# print(filelist)

# target_rootpath = 'datasets/train_add/target'
# oring_rootpath_1 = 'datasets/test/target/000'
# oring_rootpath_2 = 'datasets/test/target/011'
# oring_rootpath_3 = 'datasets/test/target/015'
# oring_rootpath_4 = 'datasets/test/target/020'

target_rootpath = 'datasets/train_add/input'
oring_rootpath_1 = 'datasets/test/input/000'
oring_rootpath_2 = 'datasets/test/input/011'
oring_rootpath_3 = 'datasets/test/input/015'
oring_rootpath_4 = 'datasets/test/input/020'

def makedir():
    print("======== make_dir")
    for i in range(64):
        path = os.path.join(target_rootpath, str(i).rjust(3, '0'))
        print(path)
        os.mkdir(path)
    print("======== make_dir_over")

def copy_list():
    orinpath = None
    targepath = None
    for i in range(64):
        if i%4 == 0:
            orinpath = oring_rootpath_1
            targepath = os.path.join(target_rootpath, str(i).rjust(3, '0'))
            copy(orinpath, targepath)
        elif i%4 == 1:
            orinpath = oring_rootpath_2
            targepath = os.path.join(target_rootpath, str(i).rjust(3, '0'))
            copy(orinpath, targepath)
        elif i%4 == 2:
            orinpath = oring_rootpath_3
            targepath = os.path.join(target_rootpath, str(i).rjust(3, '0'))
            copy(orinpath, targepath)
        else:
            orinpath = oring_rootpath_4
            targepath = os.path.join(target_rootpath, str(i).rjust(3, '0'))
            copy(orinpath, targepath)
    # 获取文件夹里面内容
def copy(orinpath, targepath):
    filelist = os.listdir(orinpath)
    # 遍历列表
    for file in filelist:
        # 拼接路径
        path = os.path.join(orinpath, file)
        tar_path = targepath
        # 不是文件夹则直接进行复制
        with open(path, 'rb') as rstream:
            container = rstream.read()
            path1 = os.path.join(tar_path, file)
            with open(path1, 'wb') as wstream:
                wstream.write(container)
    else:
        print('{} == > {} 复制完成！'.format(orinpath, targepath))


# 调用copy
# copy(src_path, target_path)

if __name__ == "__main__":
    makedir()
    copy_list()