import os
import cv2
import numpy as np

#--------------------------------------------------------#
#                       tif--->jpg
#--------------------------------------------------------#

source_dir = 'C:\\Users\\Administrator\\Desktop\\wheel\\Labels'  # 源tiff图路径
target_dir = 'C:\\Users\\Administrator\\Desktop\\wheel\\output'  # 保存到的jgp路径

# 如果目标目录不存在的话，进行目录的新建
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
files = os.listdir(source_dir)

for image_file in files:
    name = []
    portion = os.path.splitext(image_file)  # 把文件名拆分为名字和后缀
    if portion[1] == ".tif":
        name = portion[0]
        image_path = source_dir + "\\" + image_file
        image_name = target_dir + "\\" + name + ".jpg"

        # 读取tiff图片（我的全是单通道，如果需要可以写if判断一下是单还是三通道）
        img = cv2.imread(os.path.join(source_dir, image_file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image = np.zeros_like(img)
        image[:, :, 0] = gray
        image[:, :, 1] = gray
        image[:, :, 2] = gray

        # tiff转jpg
        image = image / image.max()  # 使其所有值不大于一
        image = image * 255 - 0.001  # 减去0.001防止变成负整型
        image = image.astype(np.uint8)  # 强制转换成8位整型
        b = image[:, :, 0]  # 读取蓝通道
        g = image[:, :, 1]  # 读取绿通道
        r = image[:, :, 2]  # 读取红通道
        bgr = cv2.merge([r, g, b])  # 通道拼接
        cv2.imwrite(image_name, bgr)  # 图片存储
        print("finish" + " " + image_name)