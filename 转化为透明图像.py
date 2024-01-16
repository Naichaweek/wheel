import cv2
import numpy as np
import os.path as op

def luckyFun(img1):
    mask = np.zeros(img1.shape[:2], dtype=np.uint8)
    circle = cv2.circle(mask, (1000, 900), 626, (255, 255, 255), -1)
    result = cv2.bitwise_and(img1, img1, mask=circle)
    alpha = np.ones(img1.shape[:2], dtype=np.uint8) * 255
    alpha[circle == 0] = 0
    result = cv2.merge((result[:, :, 0], result[:, :, 1], result[:, :, 2], alpha))
    return result

img_root_dir = "D:/Pycharm WorkSpace/swin_transformer/4.13/images4"

img1 = cv2.imread(op.join(img_root_dir, 'WZ-R-LD.jpg'), cv2.IMREAD_UNCHANGED)
img2 = cv2.imread(op.join(img_root_dir, 'WZ-R-LL.jpg'), cv2.IMREAD_UNCHANGED)
img3 = cv2.imread(op.join(img_root_dir, 'WZ-R-LM.jpg'), cv2.IMREAD_UNCHANGED)
img4 = cv2.imread(op.join(img_root_dir, 'WZ-R-LU.jpg'), cv2.IMREAD_UNCHANGED)
img5 = cv2.imread(op.join(img_root_dir, 'WZ-R-M.jpg'), cv2.IMREAD_UNCHANGED)
img6 = cv2.imread(op.join(img_root_dir, 'WZ-R-RD.jpg'), cv2.IMREAD_UNCHANGED)
img7 = cv2.imread(op.join(img_root_dir, 'WZ-R-RM.jpg'), cv2.IMREAD_UNCHANGED)
img8 = cv2.imread(op.join(img_root_dir, 'WZ-R-RR.jpg'), cv2.IMREAD_UNCHANGED)
img9 = cv2.imread(op.join(img_root_dir, 'WZ-R-RU.jpg'), cv2.IMREAD_UNCHANGED)

cv2.imwrite(op.join(img_root_dir, 'WZ-R-LD.png'), luckyFun(img1))
cv2.imwrite(op.join(img_root_dir, 'WZ-R-LL.png'), luckyFun(img2))
cv2.imwrite(op.join(img_root_dir, 'WZ-R-LM.png'), luckyFun(img3))
cv2.imwrite(op.join(img_root_dir, 'WZ-R-LU.png'), luckyFun(img4))
cv2.imwrite(op.join(img_root_dir, 'WZ-R-M.png'), luckyFun(img5))
cv2.imwrite(op.join(img_root_dir, 'WZ-R-RD.png'), luckyFun(img6))
cv2.imwrite(op.join(img_root_dir, 'WZ-R-RM.png'), luckyFun(img7))
cv2.imwrite(op.join(img_root_dir, 'WZ-R-RR.png'), luckyFun(img8))
cv2.imwrite(op.join(img_root_dir, 'WZ-R-RU.png'), luckyFun(img9))
