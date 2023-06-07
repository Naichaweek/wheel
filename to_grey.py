import os
from PIL import Image

#--------------------------------------------------------#
#               VOC数据集在图像分割中需要
#              jpg转灰度图之后，再转2value
#--------------------------------------------------------#

input_folder = 'C:\\Users\\Administrator\\Desktop\\VOC2007\\SegmentationClass'
output_folder = 'C:\\Users\\Administrator\\Desktop\\VOC2007\\Out'

# 遍历输入文件夹中的所有jpg文件png
for filename in os.listdir(input_folder):
    if filename.endswith('.png'):
        # 打开jpg文件并将其转换为灰度图像
        img = Image.open(os.path.join(input_folder, filename)).convert('L')

        # 将图片保存到输出文件夹中
        gray_filename = os.path.splitext(filename)[0] + '.jpg'
        img.save(os.path.join(output_folder, gray_filename))