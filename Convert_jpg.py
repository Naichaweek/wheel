import os
import numpy as np
from PIL import Image
from tqdm import tqdm

#--------------------------------------------------------#
#   该文件用于调整输入彩色图片的后缀
#--------------------------------------------------------#

Origin_JPEGImages_path   = "E:\\Deep_learning_model\\unet-pytorch\\Medical_Datasets\\Images"
Out_JPEGImages_path      = "out_jpg"

if __name__ == "__main__":
    if not os.path.exists(Out_JPEGImages_path):
        os.makedirs(Out_JPEGImages_path)

    #---------------------------#
    #   遍历标签并赋值
    #---------------------------#
    image_names = os.listdir(Origin_JPEGImages_path)
    print("正在遍历全部图片。")
    for image_name in tqdm(image_names):
        image   = Image.open(os.path.join(Origin_JPEGImages_path, image_name))
        image   = image.convert('RGB')
        image.save(os.path.join(Out_JPEGImages_path, os.path.splitext(image_name)[0] + '.jpg'))