import os
import SimpleITK as sitk
import cv2
import pydicom
from PIL import Image
#coding=utf-8
import numpy as np

def dicom2png_simple(dimage):
    img = sitk.ReadImage(dimage)
    # rescale intensity range from [-1000,1000] to [0,255]
    img = sitk.IntensityWindowing(img, -1000, 1000, 0, 255)
    # convert 16-bit pixels to 8-bit
    img = sitk.Cast(img, sitk.sitkUInt8)

    sitk.WriteImage(img, "simple_dcm2png.png")
    return img

def dicom_to_png(dicom_path, output_path):
    img = sitk.ReadImage(dicom_path)
    # rescale intensity range from [-1000,1000] to [0,255]
    img = sitk.IntensityWindowing(img, -1000, 1000, 0, 255)
    # convert 16-bit pixels to 8-bit
    img = sitk.Cast(img, sitk.sitkUInt8)

    sitk.WriteImage(img, output_path)
    return img
def dicom_batch_to_png(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # List all DICOM files in the input folder
    dicom_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.dcm')]

    for dicom_file in dicom_files:
        dicom_path = os.path.join(input_folder, dicom_file)
        output_path = os.path.join(output_folder, f"{os.path.splitext(dicom_file)[0]}.png")

        dicom_to_png(dicom_path, output_path)

def nii2png_single(nii_path, IsData = True):
    ori_data = sitk.ReadImage(nii_path)  # 读取一个数据
    data1 = sitk.GetArrayFromImage(ori_data)  # 获取数据的array
    if IsData:  #过滤掉其他无关的组织，标签不需要这步骤
        data1[data1 > 250] = 250
        data1[data1 < -250] = -250
    img_name = os.path.split(nii_path)  #分离文件名
    img_name = img_name[-1]
    img_name = img_name.split('.')
    img_name = img_name[0]
    i = data1.shape[0]
    png_path = r'C:\Users\hpc\Desktop\data\hp_pet_output'   #图片保存位置
    if not os.path.exists(png_path):
        os.makedirs(png_path)
    for j in range(0, i):   #将每一张切片都转为png
        if IsData:  # 数据
            #归一化
            slice_i = (data1[j, :, :] - data1[j, :, :].min()) / (data1[j, :, :].max() - data1[j, :, :].min()) * 255
            cv2.imwrite("%s/%s-%d.png" % (png_path, img_name, j), slice_i)  #保存
        else:   # 标签
            slice_i = data1[j, :, :] * 122
            cv2.imwrite("%s/%s-%d.png" % (png_path, img_name, j), slice_i)  # 保存

def dicom_to_images(input_folder, output_folder):
    # 如果输出文件夹不存在，则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 读取DICOM文件列表
    dicom_files = [f for f in os.listdir(input_folder) if f.endswith('.dcm')]

    # 遍历DICOM文件并转换为图片
    for dicom_file in dicom_files:
        dicom_path = os.path.join(input_folder, dicom_file)
        ds = pydicom.dcmread(dicom_path)

        # 检查是否存在像素数据
        if 'PixelData' in ds:
            # 获取像素数据
            pixel_array = ds.pixel_array

            # 创建Image对象
            img = Image.fromarray(pixel_array)

            # 保存为PNG格式（也可以选择其他格式）
            output_path = os.path.join(output_folder, f"{dicom_file[:-4]}.png")
            img.save(output_path)

            print(f"转换 {dicom_file} 到 {output_path}")
        else:
            print(f"警告：在 {dicom_file} 中未找到像素数据。")


def nii2dcm_single(nii_path, IsData = True):
    save_folder = r'C:\Users\hpc\Desktop\data\nii_output'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    ori_data = sitk.ReadImage(nii_path)  # 读取一个数据
    data1 = sitk.GetArrayFromImage(ori_data)  # 获取数据的array
    # if IsData:  # 过滤掉其他无关的组织，标签不需要这步骤
    #     data1[data1 > 250] = 250
    #     data1[data1 < -250] = -250
    data1 = np.clip(data1, -250, 250)

    img_name = os.path.split(nii_path)  #分离文件名
    img_name = img_name[-1]
    img_name = img_name.split('.')
    img_name = img_name[0]
    i = data1.shape[0]
    # 关键部分
    castFilter = sitk.CastImageFilter()
    castFilter.SetOutputPixelType(sitk.sitkInt16)
    for j in range(0, i):   #将每一张切片都转为png
        if IsData:  # 数据
            slice_i = data1[j, :, :]
            data_img = sitk.GetImageFromArray(slice_i)
            # Convert floating type image (imgSmooth) to int type (imgFiltered)
            data_img = castFilter.Execute(data_img)
            sitk.WriteImage(data_img, "%s/%s-%d.dcm" % (save_folder, img_name, j))
        else:   # 标签
            slice_i = data1[j, :, :] * 122
            label_img = sitk.GetImageFromArray(slice_i)
            # Convert floating type image (imgSmooth) to int type (imgFiltered)
            label_img = castFilter.Execute(label_img)
            sitk.WriteImage(label_img, "%s/%s-%d.dcm" % (save_folder, img_name, j))



if __name__ == "__main__":
    # 输入文件夹路径
    input_folder = r'C:\Users\hpc\Desktop\DATASET_MCL\images_PET_dcm\23'
    # 输出文件夹路径
    output_folder = r'C:\Users\hpc\Desktop\data\nii_output'
    # 单个nii文件路径 [修改函数内的保存路径]
    nii_single = r'C:\Users\hpc\Desktop\image0009.nii.gz'    # 单dcm 2 png

    # dicom2png_simple(r'C:\Users\hpc\Desktop\1.2.840.113704.1.111.2412.1677120814.24\1.2.840.113704.1.111.2584.1677120896.4336.DCM')
    # 批量dcm 2 png
    # dicom_batch_to_png(input_folder, output_folder)
    # 单个nii 2 png  数据
    nii2png_single(nii_single, IsData=True)
    # 单个nii 2 png  标签
    # nii2png_single(nii_single, IsData=False)

    # dicom_to_images(input_folder, output_folder)

    # nii2dcm_single(nii_single, True)



