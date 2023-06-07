import os
from PIL import Image

#--------------------------------------------------------#
#                      jpg-->png
#--------------------------------------------------------#
folder_path = "C:\\Users\\Administrator\\Desktop\\unet-dataset\\image"
save_path = "C:\\Users\\Administrator\\Desktop\\unet-dataset\\image_out"

# 创建保存目录
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 判断文件是否为jpg格式
    if filename.endswith(".jpg"):
        # 打开图片并转换为png格式
        img = Image.open(os.path.join(folder_path, filename))
        # 保存为png格式
        png_filename = os.path.splitext(filename)[0] + ".png"
        img.save(os.path.join(save_path, png_filename))
