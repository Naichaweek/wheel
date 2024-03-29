from PIL import Image
from pylab import *
from PCV.localdescriptors import sift
from PCV.localdescriptors import harris

# 添加中文字体支持
from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"c:/windows/fonts/SimSun.ttc", size=14)

imname = 'D:/ComputerVision_code/img/sdl11.jpg' // 需要拼接的图像的位置
im = array(Image.open(imname).convert('L'))
sift.process_image(imname, 'empire.sift') // 生成名称为empire的sift文件
l1, d1 = sift.read_features_from_file('empire.sift')

figure()
gray()
subplot(121)
sift.plot_features(im, l1, circle=False)
title(u'SIFT特征', fontproperties=font)

# 检测harris角点
harrisim = harris.compute_harris_response(im)

subplot(122)
filtered_coords = harris.get_harris_points(harrisim, 6, 0.1)
imshow(im)
plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords], '*')
axis('off')
title(u'Harris角点', fontproperties=font)
