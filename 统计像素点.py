from PIL import Image
import os
from collections import defaultdict
from tqdm import tqdm


# 定义一个函数来统计像素个数
def count_pixel_colors(image):
    pixel_counts = defaultdict(int)
    pixels = list(image.getdata())

    for pixel in pixels:
        pixel_counts[pixel] += 1

    return pixel_counts


# 指定文件夹路径
folder_path = r'C:\Users\hpc\Desktop\data\new_label_output'  # 请替换为你的文件夹路径

# 创建一个总的像素统计字典
total_pixel_counts = defaultdict(int)

# 获取PNG文件列表
png_files = [filename for filename in os.listdir(folder_path) if filename.endswith('.png')]

# 使用tqdm创建进度条
with tqdm(total=len(png_files), unit="image") as pbar:
    for filename in png_files:
        image_path = os.path.join(folder_path, filename)
        image = Image.open(image_path)

        # 统计像素个数
        pixel_counts = count_pixel_colors(image)

        # 将每个文件的像素统计结果累积到总的结果中
        for pixel, count in pixel_counts.items():
            total_pixel_counts[pixel] += count

        pbar.update(1)  # 更新进度条

# 输出总的结果
print("-" * 37)
print("|{:^15s}|{:^15s}|".format("Key", "Value"))
print("-" * 37)
for pixel, count in total_pixel_counts.items():
    print(f"|{str(pixel):^15}|{str(count):^15}|")
print("-" * 37)
