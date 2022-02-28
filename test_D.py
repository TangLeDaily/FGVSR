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


import shutil
# target  input
path = 'datasets/train_add/target/015/'


def test():
    for i in range(50):
        oldName = os.path.join(path,str(50+i).rjust(8, '0')+'.png')
        newName = os.path.join(path,str(i).rjust(8, '0')+'.png')
        shutil.copy(oldName, newName)
        print(i)

# 调用copy
# copy(src_path, target_path)

if __name__ == "__main__":
    test()