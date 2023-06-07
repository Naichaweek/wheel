import cv2
import os

#--------------------------------------------------------#
#                       tif--->png
#--------------------------------------------------------#

tif_path = 'Labels'
jpg_path = 'outtttt'

imgs = os.listdir(tif_path)

for i, img in enumerate(imgs):
    img_name = os.path.join(tif_path, img)

    file = cv2.imread(img_name, 1)

    save_file = os.path.join(jpg_path, img.strip('.tif') + '.png')

    cv2.imwrite(save_file, file)

    print(i)
