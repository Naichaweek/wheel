from PIL import Image
#
# # 创建一个全黑色的图片，尺寸为200x200
# image = Image.new("RGB", (4000, 3200), color="black")
# img1 = Image.open("testv/v1/css-R-RU.jpg")
# image.paste(img1,(1000,800))
# image.save("css-R-RU1.jpg")
# # image_resized = image.resize((2000, 1600))
# # image_resized.save("css-R-LL1.jpg")

import os
import sys
import cv2
import numpy

# 需要处理的原始图片的路径
img_root_dir = "C:/Users/Administrator/Desktop/wheel/pic_after_enhance/"

def main():
    for img_dir in os.listdir(img_root_dir):
        temp_dir = img_root_dir + img_dir + "/"
        for img_path in os.listdir(temp_dir):
            pic_path = temp_dir + img_path

            image = Image.new("RGB", (4000, 3200), color="black")
            img1 = Image.open(pic_path)
            image.paste(img1,(1000,800))
            image.save(pic_path+"css.jpg")
            # image_resized = image.resize((2000, 1600))
            # image_resized.save("css-R-LL1.jpg")

if __name__ == '__main__':
    main()