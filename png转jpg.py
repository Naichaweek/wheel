import os
from PIL import Image

def convert_png_to_jpg(input_folder, output_folder):
    # 创建目标文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历原始文件夹中的所有PNG文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            # 打开PNG文件
            png_path = os.path.join(input_folder, filename)
            png_image = Image.open(png_path)

            # 构建目标JPEG文件路径
            jpg_filename = os.path.splitext(filename)[0] + '.jpg'
            jpg_path = os.path.join(output_folder, jpg_filename)

            # 将PNG图像保存为JPEG格式
            png_image.convert('RGB').save(jpg_path, 'JPEG')

            print(f'{png_path} -> {jpg_path}')

if __name__ == '__main__':
    # 输入原始文件夹和目标文件夹的路径
    input_folder = 'TIF_to_png_out'
    output_folder = 'output_jpg_images'

    # 将PNG文件转换为JPEG
    convert_png_to_jpg(input_folder, output_folder)