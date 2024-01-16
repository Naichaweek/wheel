import SimpleITK
from PIL import Image
import gzip
import shutil

from matplotlib import pyplot as plt

# 定义文件路径
#CT
# nii_file_path = r"C:\Users\hpc\Desktop\data\train\image\CT\2.nii.gz"
#PET
# nii_file_path = r"C:\Users\hpc\Desktop\data\train\image\PET\2.nii.gz"
# label
# nii_file_path = r"C:\Users\hpc\Desktop\data\train\label\2.nii.gz"

# example
# ct
# nii_file_path = r"C:\Users\hpc\Desktop\archive\Hippocampus\imagesTr\hippocampus_001.nii"
# label
# nii_file_path = r"C:\Users\hpc\Desktop\archive\Hippocampus\labelsTr\hippocampus_001.nii"

# ct
nii_file_path = r"C:\Users\hpc\Desktop\DATASET_Synapse\unetr_pp_raw\unetr_pp_raw_data\Task02_Synapse\imagesTr\img0001.nii.gz"
# label
# nii_file_path = r"C:\Users\hpc\Desktop\data\out1.nii.gz"
#
# 使用 SimpleITK 读取未压缩的 NIfTI 文件
ct_img = SimpleITK.ReadImage(nii_file_path)
ct_array = SimpleITK.GetArrayFromImage(ct_img)
print(ct_array.shape)

# 选择要可视化的切片索引
slice_index = 100
img = ct_array[slice_index, :, :]

# 打印切片形状
print(img.shape)

# 将切片转换为图像并可视化
img_pic = Image.fromarray(img)
plt.imshow(img_pic, cmap='gray')
# plt.axis('off')
# plt.imshow(img_pic)
plt.show()






#
# import SimpleITK
# from PIL import Image
# import gzip
# import shutil
# from matplotlib import pyplot as plt
# import numpy as np
#
# # 定义二个文件路径
# nii_file_path1 = r"C:\Users\hpc\Desktop\data\1.nii.gz"
# nii_file_path2 = r"C:\Users\hpc\Desktop\data\out3.nii.gz"
#
# # 使用 SimpleITK 读取未压缩的 NIfTI 文件
# ct_img1 = SimpleITK.ReadImage(nii_file_path1)
# ct_array1 = SimpleITK.GetArrayFromImage(ct_img1)
#
# # 使用 SimpleITK 读取第二个未压缩的 NIfTI 文件
# ct_img2 = SimpleITK.ReadImage(nii_file_path2)
# ct_array2 = SimpleITK.GetArrayFromImage(ct_img2)
#
# # 选择要可视化的切片索引
# slice_index = 135
#
# # 获取切片
# img1 = ct_array1[slice_index, :, :]
# img2 = ct_array2[slice_index, :, :]
#
# # 创建两个子图
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
#
# # 在每个子图中显示一个切片
# ax1.imshow(img1, cmap='gray')
# ax1.set_title('Origin')
# ax1.axis('on')
#
# ax2.imshow(img2, cmap='gray')
# ax2.set_title('fix')
# ax2.axis('on')
#
# plt.show()