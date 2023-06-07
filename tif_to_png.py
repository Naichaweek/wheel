from PIL import Image
import os

#--------------------------------------------------------#
#                       tif--->png
#--------------------------------------------------------#

dir_path = "Labels"
save_path = "test"

# 如果保存目录不存在，则创建该目录
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 遍历目录下的所有.tif文件
for file_name in os.listdir(dir_path):
    if file_name.endswith(".tif"):
        # 打开.tif文件
        img = Image.open(os.path.join(dir_path, file_name))
        # 构造新的文件名和路径
        new_name = os.path.splitext(file_name)[0] + ".png"
        new_path = os.path.join(save_path, new_name)
        # 保存为.png文件
        img.save(new_path)
