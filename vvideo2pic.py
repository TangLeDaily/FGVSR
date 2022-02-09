import cv2 as cv
import os.path

def img_resize(image):
    height, width = image.shape[0], image.shape[1]
    # 设置新的图片分辨率框架
    width_new = 1280
    height_new = 720
    # 判断图片的长宽比率
    if width / height >= width_new / height_new:
        img_new = cv.resize(image, (width_new, int(height * width_new / width)))
    else:
        img_new = cv.resize(image, (int(width * height_new / height), height_new))
    img_lr = cv.resize(img_new, (int(width * height_new / height // 4), height_new // 4))
    return img_new, img_lr
def int_to_str(i):
    if i<10:
        return "000{}".format(i)
    elif i<100:
        return "00{}".format(i)
    elif i<1000:
        return "0{}".format(i)
    else:
        return str(i)
def saveimg(lr, hr, name):
    nam = int_to_str(name)+".png"
    cv.imwrite("datasets/test/Videoget/input/001/" + nam, lr)
    cv.imwrite("datasets/test/Videoget/target/001/" + nam, hr)

def save(img, name):
    hr, lr = img_resize(img)
    saveimg(lr, hr, name)


def video_to_pic():
    # 1.存储图片文件夹
    path = 'datasets/test/Videoget/A'  # 存放视频图片的主目录
    if not os.path.exists(path):  # 如果不存在就创建文件夹
        os.mkdir(path)

    # 2.读取视频文件夹
    filepath = 'datasets/video'  # 需要读取的视频的路径
    pathDir = os.listdir(filepath)  # 获取文件夹中文件名称

    # 3.截视频帧数
    for allDir in pathDir:  # 逐个读取视频文件
        a = 1  # 图片计数-不改
        c = 1  # 帧数计数-不改
        videopath = r'datasets/video/' + allDir  # 视频文件路径
        vc = cv.VideoCapture(videopath)  # 读入视频文件
        # 存储视频的子目录
        path = 'datasets/test/Videoget/A/' + allDir.split('.')[0]
        if not os.path.exists(path):  # 如果不存在就创建文件夹
            os.mkdir(path)

        if vc.isOpened():
            rval, frame = vc.read()
        else:
            rval = False

        timeF = 4  # 帧数间隔
        while rval:
            rval, frame = vc.read()  # 分帧读取视频
            if rval == False:
                break
            if (c % timeF == 0):
                # cv.imwrite(path + '/' + str(a) + '.jpg', frame)  # 保存路径
                save(frame, a)
                a = a + 1
            c = c + 1
            cv.waitKey(1)
        vc.release()

if __name__ == "__main__":
    video_to_pic()