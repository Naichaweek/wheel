import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

# 读取NIfTI文件
input_path = r"C:\Users\hpc\Desktop\data\ctimg0023.nii.gz"
output_path = r"C:\Users\hpc\Desktop\data\small_ctimg1.nii.gz"

# img = nib.load(input_path)
# data = img.get_fdata()
#
# # 计算缩放比例
# scale_factors = (
#     data.shape[0] / 144,
#     data.shape[1] / 144
# )
#
# # 缩小图像数据
# resized_data = np.zeros((144, 144, data.shape[2]), dtype=data.dtype)
# for z in range(data.shape[2]):
#     # 获取每个切片
#     slice_data = data[:, :, z]
#
#     # 缩小切片
#     resized_slice = np.zeros((144, 144), dtype=data.dtype)
#     for i in range(144):
#         for j in range(144):
#             x = int(i * scale_factors[0])
#             y = int(j * scale_factors[1])
#             resized_slice[i, j] = slice_data[x, y]
#
#     resized_data[:, :, z] = resized_slice
#
# # 创建新的NIfTI图像
# resized_img = nib.Nifti1Image(resized_data, img.affine)
# nib.save(resized_img, output_path)

img = nib.load(input_path)
data = img.get_fdata()

# 计算缩放比例
scale_factors = (
    144 / data.shape[0],
    144 / data.shape[1],
    1  # 不缩放切片数量
)

# 使用三次样条插值缩小图像数据
resized_data = zoom(data, scale_factors, order=3)

# 创建新的NIfTI图像
resized_img = nib.Nifti1Image(resized_data, img.affine)
nib.save(resized_img, output_path)