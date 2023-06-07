from PIL import Image
import numpy as np

#--------------------------------------------------------#
#              读取指定灰度图，非0的像素值显示出来
#--------------------------------------------------------#

image = Image.open('output_2value/0001.png')  # 图片的路径

# #--------------所有非0都【重复】显示出来-------------------#
# a, b = image.size  # 获得图像的长、宽
# for i in range(a):  # 遍历图像的行
#     for j in range(b):  # 遍历图像的列
#         pixel = image.getpixel((i, j))  # 读取该点的像素值
#         if pixel != 0:
#             print('pixel:', pixel)


#--------------所有非0都【不重复】显示出来-------------------#
a, b = image.size  # 获得图像的长、宽
non_zero_pixels = set()

for i in range(a):  # 遍历图像的行
    for j in range(b):  # 遍历图像的列
        pixel = image.getpixel((i, j))  # 读取该点的像素值
        if pixel != 0:
            non_zero_pixels.add(pixel)
print(f"图像中非0像素值有：{non_zero_pixels}")
