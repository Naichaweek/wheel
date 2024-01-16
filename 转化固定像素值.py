import os
from PIL import Image

def convert_jpg_to_png_with_mask(input_folder, output_folder):
    # 创建目标文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历原始文件夹中的所有JPEG文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            # 打开JPEG文件
            jpg_path = os.path.join(input_folder, filename)
            jpg_image = Image.open(jpg_path)

            # 创建一个新的图像对象
            modified_image = Image.new('RGB', jpg_image.size)
            width, height = modified_image.size

            # 将非(0, 0, 0)的像素值变成(128, 0, 0)
            for x in range(width):
                for y in range(height):
                    pixel = jpg_image.getpixel((x, y))
                    if pixel != (0, 0, 0):
                        modified_image.putpixel((x, y), (1, 0, 0))
                    else:
                        modified_image.putpixel((x, y), pixel)
                    # if pixel == (255, 255, 255):
                    #     modified_image.putpixel((x, y), (0, 0, 0))
                    # else:
                    #     modified_image.putpixel((x, y), pixel)


            # 构建目标PNG文件路径
            png_filename = os.path.splitext(filename)[0] + '.png'
            png_path = os.path.join(output_folder, png_filename)

            # 保存修改后的图像为PNG格式
            modified_image.save(png_path, 'PNG')

            print(f'{jpg_path} -> {png_path}')

if __name__ == '__main__':
    # 输入原始文件夹和目标文件夹的路径
    input_folder = r'C:\Users\hpc\Desktop\train'
    output_folder = r'C:\Users\hpc\Desktop\train_g'

    # 将 png 文件转换为 mask
    convert_jpg_to_png_with_mask(input_folder, output_folder)