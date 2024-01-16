import os
import re
import shutil

# 提供的文件夹路径
source_folder = r"C:\Users\hpc\Desktop\labelsTr_d"

# 保存的文件夹路径
destination_folder = r"C:\Users\hpc\Desktop\lre"

# 获取文件列表
file_list = [f for f in os.listdir(source_folder) if f.endswith(".nii.gz")]

# 自定义排序函数
def custom_sort(file_name):
    match = re.search(r'labels(\d+).nii.gz', file_name)
    if match:
        # 提取文件名中的数字部分
        number_part = int(match.group(1))
        return number_part
    else:
        # 如果提取失败，返回一个足够大的数字确保放在列表的最后
        return float('inf')

# 按照自定义排序函数排序文件列表
sorted_file_list = sorted(file_list, key=custom_sort)

# 创建保存文件夹
os.makedirs(destination_folder, exist_ok=True)

# 复制文件到保存文件夹，并重新命名
for index, file_name in enumerate(sorted_file_list, start=1):
    source_path = os.path.join(source_folder, file_name)
    new_file_name = f"labels{index:04d}.nii.gz"
    destination_path = os.path.join(destination_folder, new_file_name)

    try:
        shutil.copyfile(source_path, destination_path)
        print(f"Copying {file_name} to {destination_folder} as {new_file_name}")
    except Exception as e:
        print(f"Error copying {file_name} to {destination_folder}: {e}")
