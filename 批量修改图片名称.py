import os
import cv2

img_root = "MA"  # 图片地址

# 遍历文件夹中的所有文件
for filename in os.listdir(img_root):
    if filename.endswith('_MA.tif'):
        # 构建新的文件名
        new_filename = filename.replace('_MA.tif', '.tif')

        # 构建原始文件的完整路径和新文件的完整路径
        old_filepath = os.path.join(img_root, filename)
        new_filepath = os.path.join(img_root, new_filename)

        # 使用os.rename()函数重命名文件
        os.rename(old_filepath, new_filepath)

print("批量修改完成。")