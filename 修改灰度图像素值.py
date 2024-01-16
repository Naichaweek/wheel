import os
import cv2
from tqdm import tqdm

#--------------------------------------------------------#
#         读取指定灰度图，非0的像素值显示出来
#--------------------------------------------------------#

# img_dir = r'C:\Users\hpc\Desktop\e_ophtha_MA\e_optha_MA\SegmentationClass'
# save_dir = r'C:\Users\hpc\Desktop\e_ophtha_MA\e_optha_MA\out'
# img_files = os.listdir(img_dir)  # 获取所有文件名
#
# # 循环遍历文件夹中的所有图片
# for img_file in tqdm(img_files, desc='处理进度'):
#     img_path = os.path.join(img_dir, img_file)
#     # 读取灰度图
#     img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#     # 循环遍历灰度图中的每一个像素
#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             if img[i, j] == 255:
#                 img[i, j] = 1
#             # if img[i, j] == 38:
#             #     img[i, j] = 2
#     # 保存修改后的图片
#     new_img_path = os.path.join(save_dir, img_file)
#     cv2.imwrite(new_img_path, img)
#
import os
import cv2
import numpy as np
from tqdm import tqdm

# --------------------------------------------------------#
#         读取指定灰度图，非0的像素值显示出来
# --------------------------------------------------------#

img_dir = r'C:\Users\hpc\Desktop\data\hp_nii_output'
save_dir = r'C:\Users\hpc\Desktop\data\cov_nii_out'
img_files = os.listdir(img_dir)  # 获取所有文件名

# 循环遍历文件夹中的所有图片
for img_file in img_files:
    img_path = os.path.join(img_dir, img_file)
    # 读取灰度图
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # 检查图像是否成功加载
    if img is not None:
        # 将非零像素值设置为255
        img[img != 0] = 255

        # 保存修改后的图片
        new_img_path = os.path.join(save_dir, img_file)
        cv2.imwrite(new_img_path, img)
    else:
        print(f"无法读取图像：{img_path}")