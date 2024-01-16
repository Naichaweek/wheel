import os
import random
import shutil

# 设置随机种子，确保每次运行结果一致
random.seed(42)

# 指定图片文件夹的路径
image_folder = r'C:\Users\hpc\Desktop\VOCdevkit\VOC2007\JPEGImages'

# 创建输出文件夹
output_folder = r'C:\Users\hpc\Desktop\VOCdevkit\VOC2007\txt_out'
os.makedirs(output_folder, exist_ok=True)

# 获取图片文件夹中所有图片的文件名
image_files = sorted(os.listdir(image_folder))

# 将所有图片的文件名保存到trainval.txt文件中
with open(os.path.join(output_folder, 'trainval.txt'), 'w') as trainval_file:
    trainval_file.writelines([os.path.splitext(name)[0] + '\n' for name in image_files])

# 设置划分比例，例如，这里将80%的图片分为训练集，20%分为验证集
split_rate = 0.9

# 随机打乱图片文件列表
random.shuffle(image_files)

# 计算划分的索引位置
split_index = int(len(image_files) * split_rate)

# 分割训练集和验证集
train_files = image_files[:split_index]
val_files = image_files[split_index:]

# 将训练集图片文件名保存到train.txt文件中
with open(os.path.join(output_folder, 'train.txt'), 'w') as train_file:
    train_file.writelines([os.path.splitext(name)[0] + '\n' for name in train_files])

# 将验证集图片文件名保存到val.txt文件中
with open(os.path.join(output_folder, 'val.txt'), 'w') as val_file:
    val_file.writelines([os.path.splitext(name)[0] + '\n' for name in val_files])

print("划分完成并保存到输出文件夹。")