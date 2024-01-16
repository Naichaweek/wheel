import os
import shutil


def add_suffix_to_images(input_folder, output_folder, suffix):
    if not os.path.exists(input_folder):
        print("输入文件夹路径不存在")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # 如果输出文件夹不存在，创建它

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # 获取文件名和后缀名
            name, ext = os.path.splitext(filename)

            # 添加后缀并转换为小写
            new_filename = name + suffix + ext.lower()

            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, new_filename)

            shutil.copyfile(input_path, output_path)  # 复制文件到新文件夹
            print(f"复制：{filename} -> {output_path}")

    print("复制完成，开始重命名...")

    # 重命名新文件夹中的文件
    for filename in os.listdir(output_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            name, ext = os.path.splitext(filename)
            new_filename = name + suffix + ext.lower()

            old_path = os.path.join(output_folder, filename)
            new_path = os.path.join(output_folder, new_filename)

            os.rename(old_path, new_path)
            print(f"重命名：{filename} -> {new_filename}")


input_folder = r"C:\Users\hpc\Desktop\VOCdevkit\VOC2007\JPEGImages"  # 替换为你的输入文件夹路径
output_folder = r"C:\Users\hpc\Desktop\VOCdevkit\VOC2007\JPEGImages1"  # 替换为你的输出文件夹路径
suffix = ""

add_suffix_to_images(input_folder, output_folder, suffix)
