import os
from PIL import Image

#--------------------------------------------------------#
#                 将图片转化成二值图像
#--------------------------------------------------------#

def convert_to_mask(input_folder, output_folder, threshold=0):#设置阈值
    # 创建目标文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历原始文件夹中的所有 tif 文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg'):
            # 打开 tif 文件
            tif_path = os.path.join(input_folder, filename)
            tif_image = Image.open(tif_path)

            # 将 tif 文件转换为二值图像
            mask_image = tif_image.convert('L').point(lambda x: 1 if x > threshold else 0)
            # 保存 mask 文件
            mask_filename = filename
            mask_path = os.path.join(output_folder, mask_filename)
            mask_image.save(mask_path)

            print(f'{tif_path} -> {mask_path}')

if __name__ == '__main__':
    # 输入原始文件夹和目标文件夹的路径
    input_folder = 'output_grey'
    output_folder = 'output_2value'

    # 将 tif 文件转换为 mask
    convert_to_mask(input_folder, output_folder)