import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 加载 NIfTI 文件
nii_file_path = r'C:\Users\hpc\Desktop\DATASET_MCL\images_PET\img0016.nii.gz'
label_file_path = r'C:\Users\hpc\Desktop\DATASET_MCL\labels_modify\label0016.nii.gz'

img = nib.load(nii_file_path)
data = img.get_fdata()

label_img = nib.load(label_file_path)
label_data = label_img.get_fdata()

# 将数据展平为一维数组
flat_data = data.flatten()

# 绘制归一化前的曲线图和标签像素值统计
plt.figure(figsize=(15, 5))

# 绘制归一化前的曲线图
plt.subplot(1, 3, 1)
plt.plot(sorted(flat_data), color='blue', alpha=0.7)
plt.title('Pixel Value Distribution (Before Normalization)')
plt.xlabel('Index')
plt.ylabel('Pixel Value')

# 归一化数据
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data.flatten().reshape(-1, 1)).reshape(data.shape)

# 绘制归一化后的曲线图
plt.subplot(1, 3, 2)
flat_data_normalized = data_normalized.flatten()
plt.plot(sorted(flat_data_normalized), color='orange', alpha=0.7)
plt.title('Normalized Pixel Value Distribution')
plt.xlabel('Index')
plt.ylabel('Normalized Pixel Value')

# 绘制标签像素值统计
plt.subplot(1, 3, 3)
flat_label_data = label_data.flatten()
plt.plot(sorted(flat_label_data), color='green', alpha=0.7)
plt.title('Label Pixel Value Distribution')
plt.xlabel('Index')
plt.ylabel('Label Pixel Value')

plt.tight_layout()
plt.show()
