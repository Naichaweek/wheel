import cv2 as cv
import numpy as np

def contrast_Ratio_brightness(image,a,g):
    #a为对比度，g为亮度
    image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    h,w,c=image.shape
    #创建一个全白色图像。
    mask=np.zeros([h,w,c],image.dtype)
    #cv.addWeighted函数对两张图片线性加权叠加
    dstImage=cv.addWeighted(image,a,mask,1-a,g)
    cv.imshow("dstImage",dstImage)
srcImage4=cv.imread(r'C:\Users\hpc\Desktop\data\hp_pet_output\image0009-198.png',-1)
print(srcImage4.shape)
cv.imshow("Saber",srcImage4)
contrast_Ratio_brightness(srcImage4,1.2,10)
cv.waitKey(0)
cv.destroyAllWindows()
"""
cv.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]])，计算两个数组的加权和。

src1：参数src1第一个输入数组。

alpha：第一个数组元素的参数alpha权重。

src2：参数src2与src1大小和通道号相同的第二个输入数组。

beta：第二个数组元素的参数beta权重。

gamma：将参数gamma scalar添加到每个和中。

dst：输出与输入阵列具有相同大小和通道数的dst输出阵列。

dtype：输出数组的可选深度；当两个输入数组的深度相同时，dtype可以设置为-1，相当于src1.depth（）。
"""

