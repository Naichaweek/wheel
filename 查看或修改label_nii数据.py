import nibabel as nib
import numpy as np



# # nifti_file_path = r'C:\Users\hpc\Desktop\data\label0023.nii.gz'
# source_nifti_path = r'C:\Users\hpc\Desktop\data\labela23test.nii.gz' # label0023.nii.gz副本
# # nifti_file_path = r'C:\Users\hpc\Desktop\data\newla2.nii.gz'
# target_nifti_path = r'C:\Users\hpc\Desktop\data\last_test.nii.gz'
#
#
# # 读取 NIfTI 文件
# source_nifti_img  = nib.load(source_nifti_path)
#
# # 获取图像数据
# nifti_data = source_nifti_img.get_fdata()
#
# # # 将非零值修改为 1
# nifti_data[nifti_data != 0] = 1
#
#
# # 获取唯一的像素值和它们的计数
# unique_values, counts = np.unique(nifti_data, return_counts=True)
#
# # 打印唯一的像素值和它们的计数
# for value, count in zip(unique_values, counts):
#     print(f"像素值 {value} 出现次数: {count}")
#
# # 获取头部信息
# nifti_header = source_nifti_img.header
#
# # 打印图像数据的形状和头部信息
# print("图像数据形状:", nifti_data.shape)
# print("图像数据形状:", nifti_data)
# print("头部信息:", nifti_header)
#
# # 创建 NIfTI 图像对象
# modified_nifti_img = nib.Nifti1Image(nifti_data, source_nifti_img.affine, header=source_nifti_img.header)
#
# # 保存修改后的数据为新的 NIfTI 文件
# nib.save(modified_nifti_img, target_nifti_path)

'''
此段代码只修改一个nii.gz
'''
import os
import nibabel as nib
import numpy as np
import glob
def modify_nifti_data(input_path, output_path,modify):
    # 读取 NIfTI 文件
    source_nifti_img = nib.load(input_path)

    # 获取图像数据
    nifti_data = source_nifti_img.get_fdata()

    if modify:
        # 将非零值修改为 1
        nifti_data[nifti_data != 0] = 1

        # 获取唯一的像素值和它们的计数
        unique_values, counts = np.unique(nifti_data, return_counts=True)

        # 打印唯一的像素值和它们的计数
        for value, count in zip(unique_values, counts):
            print(f"像素值 {value} 出现次数: {count}")

        # 获取头部信息
        nifti_header = source_nifti_img.header

        # 打印图像数据的形状和头部信息
        print("图像数据形状:", nifti_data.shape)
        print("图像数据形状:", nifti_data)
        print("头部信息:", nifti_header)

        # 创建 NIfTI 图像对象
        modified_nifti_img = nib.Nifti1Image(nifti_data, source_nifti_img.affine, header=nifti_header)

        # 保存修改后的数据为新的 NIfTI 文件
        nib.save(modified_nifti_img, output_path)

    else:
        # 获取唯一的像素值和它们的计数
        unique_values, counts = np.unique(nifti_data, return_counts=True)

        # 打印唯一的像素值和它们的计数
        for value, count in zip(unique_values, counts):
            print(f"像素值 {value} 出现次数: {count}")

        # 获取头部信息
        nifti_header = source_nifti_img.header

        # 打印图像数据的形状和头部信息
        print("图像数据形状:", nifti_data.shape)
        print("图像数据形状:", nifti_data)
        print("头部信息:", nifti_header)


# 使用函数
target_nifti_path = r'C:\Users\hpc\Desktop\data\test_out.nii.gz'  #修改后保存路径

source_nifti_path = r'C:\Users\hpc\Desktop\float32.nii.gz'
# modify_nifti_data(source_nifti_path, target_nifti_path,modify=False)  #是否修改modify后保存


'''
此段代码批量修改nii.gz
'''
def modify_and_save_batch(input_folder, output_folder):
    # 获取输入文件夹中的所有 NIfTI 文件
    nifti_files = glob.glob(os.path.join(input_folder, '*.nii.gz'))

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 循环处理每个 NIfTI 文件
    for nifti_path in nifti_files:
        # 构造输出路径
        file_name = os.path.basename(nifti_path)
        output_path = os.path.join(output_folder, file_name)

        # 读取 NIfTI 文件
        source_nifti_img = nib.load(nifti_path)

        # 获取图像数据
        nifti_data = source_nifti_img.get_fdata()

        # 将非零值修改为 1
        nifti_data[nifti_data != 0] = 1

        # 获取头部信息
        nifti_header = source_nifti_img.header

        # 创建 NIfTI 图像对象
        modified_nifti_img = nib.Nifti1Image(nifti_data, source_nifti_img.affine, header=nifti_header)

        # 保存修改后的数据为新的 NIfTI 文件
        nib.save(modified_nifti_img, output_path)

        print(f"已处理文件: {nifti_path}，保存为: {output_path}")

# 使用函数进行批量处理
input_folder = r'C:\Users\hpc\Desktop\gllabelsTr'
output_folder = r'C:\Users\hpc\Desktop\Task13_Mcl\labelsTr'
modify_and_save_batch(input_folder, output_folder)



