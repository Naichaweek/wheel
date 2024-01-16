import os
#
# # 指定文件夹路径
# folder_path = r'C:\Users\hpc\Desktop\qwer'  # 请替换为你的文件夹路径
# i=0
#
# # 获取文件夹下所有图片文件的名称
# image_files = [filename for filename in os.listdir(folder_path) if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tif', '.tiff'))]
#
# # 写入文件名到train.txt
# with open(r'C:\Users\hpc\Desktop\trainval.txt', 'w') as file:
#     for filename in image_files:
#         file.write(filename + '\n')
#         i = i + 1
#
# print(i)
# i = 0
# print(f"Image file names saved to train.txt")

# 指定文件夹路径
folder_path =r'C:\Users\hpc\Desktop\qwer'  # 请替换为你的文件夹路径
i=0

# 获取文件夹下所有图片文件的名称（去除后缀）
image_files = [os.path.splitext(filename)[0] for filename in os.listdir(folder_path) if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tif', '.tiff'))]

# 写入文件名（去除后缀）到 train.txt
with open(r'C:\Users\hpc\Desktop\trainval.txt', 'w') as file:
    for filename in image_files:
        file.write(filename + '\n')
        i = i + 1

print(i)
i = 0
print(f"Image file names (without extension) saved to train.txt")