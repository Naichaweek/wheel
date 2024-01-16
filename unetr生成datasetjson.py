import os
import random
import json
from shutil import copyfile
import numpy as np

# 设置固定随机种子
random.seed(42)

# 设置文件夹路径
image_folder = r"C:\Users\hpc\Desktop\glre" # 源图
label_folder = r"C:\Users\hpc\Desktop\lre" # 源标签

train_folder = r"C:\Users\hpc\Desktop\Task13_Mcl\imagesTr" # 训练集
test_folder = r"C:\Users\hpc\Desktop\Task13_Mcl\imagesTs" # 测试集

label_train_folder = r"C:\Users\hpc\Desktop\Task13_Mcl\labelsTr"# 训练标签集
dataset_json_path = r"C:\Users\hpc\Desktop\Task13_Mcl\dataset.json"
split_txt_path = r"C:\Users\hpc\Desktop\Task13_Mcl\split.txt"

# 获取所有图片和标签文件列表
image_files = [f for f in os.listdir(image_folder) if f.endswith(".nii.gz")]
label_files = [f for f in os.listdir(label_folder) if f.endswith(".nii.gz")]

# 创建测试集文件夹
os.makedirs(test_folder, exist_ok=True)

# 选择20%的图片作为测试集
num_test = int(0.2 * len(image_files))
test_indices = random.sample(range(len(image_files)), num_test)


for index in test_indices:
    img_file = image_files[index]
    copyfile(os.path.join(image_folder, img_file), os.path.join(test_folder, img_file))

# 创建训练集和标签集文件夹
os.makedirs(train_folder, exist_ok=True)
os.makedirs(label_train_folder, exist_ok=True)

# 创建dataset.json的内容
dataset_json = {
    "labels": {"0": "background", "1": "mcl"},
    "licence": "see challenge website",
    "modality": {"0": "PET"},
    "name": "Mcl",
    "numTest": num_test,
    "numTraining": len(image_files),
    "description": "It seems that we use the whole data to train, but we will split the validation set from the training set",
    "reference": "see challenge website",
    "release": "0.0",
    "tensorImageSize": "4D",
    "test": [f"./imagesTs/{img_file}" for img_file in sorted(os.listdir(test_folder))],
    "training": [
        {
            "image": f"./imagesTr/{img_file}",
            "label": f"./labelsTr/{label_file}"
        } for img_file, label_file in zip(sorted(os.listdir(train_folder)), sorted(os.listdir(label_train_folder)))
    ]
}

# 复制训练集和标签集文件
for img_file, label_file in zip(image_files, label_files):
    # 复制训练集图片
    copyfile(os.path.join(image_folder, img_file), os.path.join(train_folder, img_file))

    # 复制标签集
    copyfile(os.path.join(label_folder, label_file), os.path.join(label_train_folder, label_file))

# 将dataset.json保存到文件
with open(dataset_json_path, 'w') as json_file:
    json.dump(dataset_json, json_file, indent=4)

# Create lists to store the names of images in the validation and training sets
val_image_names = []
train_image_names = []

# Copy images to the test set and record their names
for index in range(len(image_files)):
    img_file = image_files[index]
    img_name = os.path.splitext(img_file)[0]  # Remove the file extension
    img_name = img_name.replace(".nii", "")  # Remove the "img" prefix

    if index in test_indices:
        val_image_names.append(img_name)
        copyfile(os.path.join(image_folder, img_file), os.path.join(test_folder, img_file))
    else:
        train_image_names.append(img_name)
        copyfile(os.path.join(image_folder, img_file), os.path.join(train_folder, img_file))

with open(split_txt_path, 'w') as split_file:
    split_file.write(f"val: {str(val_image_names)}\n")
    split_file.write(f"train: {str(train_image_names)}\n")