import math

import cv2
import numpy as np
import scipy.stats as stats
from PIL import Image


# 定义误差函数
def compute_error(pt1, pt2, H):
    """
    计算点对之间的重投影误差
    """
    pt1_homo = np.concatenate((pt1, np.ones((pt1.shape[0], 1))), axis=1)
    pt2_homo = np.concatenate((pt2, np.ones((pt2.shape[0], 1))), axis=1)
    pt1_transform = np.dot(H, pt1_homo.T).T
    pt2_transform = np.dot(np.linalg.pinv(H), pt2_homo.T).T
    pt1_transform_norm = pt1_transform[:, :2] / pt1_transform[:, 2:]
    pt2_transform_norm = pt2_transform[:, :2] / pt2_transform[:, 2:]
    error = np.sum(np.square(pt1_transform_norm - pt2[:, :2]), axis=1)
    return error

# 定义MLESAC算法
def mlesac(pt1, pt2, max_iterations, threshold):
    """
    使用MLESAC算法去除误匹配点
    """
    max_confidence = 0
    best_H = None
    best_inliers = None
    best_error = np.inf
    for i in range(max_iterations):
        # 随机采样内点
        indices = np.random.choice(pt1.shape[0], 4, replace=False)
        # 估计变换矩阵
        H, _ = cv2.findHomography(pt1[indices], pt2[indices], cv2.RANSAC, 5.0)
        # 计算所有点的误差
        error = compute_error(pt1, pt2, H)
        # 计算内点
        inliers = np.where(error < threshold)[0]
        # 计算置信度
        confidence = len(inliers) / pt1.shape[0]
        if confidence > max_confidence:
            max_confidence = confidence
        # 更新最佳模型
        if confidence > 0.001 and len(inliers) > 4:
            mean_error = np.mean(error[inliers])
            if mean_error < best_error:
                best_error = mean_error
                best_H = H
                best_inliers = inliers
    return best_H, best_inliers,max_confidence



def My_addWeight(img1, img2):
    for x in range(0, img1.width):
        for y in range(0, img1.height):
                r1, g1, b1 = list(img1.getpixel((x, y)))
                if r1 != 0 and g1 != 0 and b1 != 0:
                    img2.putpixel((x, y), (int(r1), int(g1), int(b1)))
    return img2



img1 = cv2.imread("testv/v1/css-R-LM.jpg")
img2 = cv2.imread("testv/v1/css-R-LL.jpg")
# cv2.Stitcher_create()
# 转换为灰度图像
img3 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img4 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# cv2.namedWindow("BFmatch", cv2.WINDOW_NORMAL)
# cv2.imshow("BFmatch", img3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # 限制对比度的自适应直方图均衡化
#
# # 对RGB三通道都进行直方图均衡化
# img3 = img_red_process_CLAHE = clahe.apply(img3)
# img4 = img_red_process_CLAHE = clahe.apply(img4)

# cv2.namedWindow("BFmatch", cv2.WINDOW_NORMAL)
# cv2.imshow("BFmatch", img3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# kps2 是关键点（keypoints）的列表，每个关键点具有一组坐标和一个尺度。
# features2是关键点的特征描述符，也可以看做是对关键点周围区域的局部描述。
# 这些特征点和描述符可以用于图像匹配、拼接、目标检测等任务。
# 在该代码中，kps2 和 features2 是通过SIFT算法从图像img2中提取的关键点和特征描述符。
# nfeatures，保留的最佳特性的数量。特征按其得分进行排序(以SIFT算法作为局部对比度进行测量)；
# nOctavelLayers，高斯金字塔最小层级数，由图像自动计算出；
# constrastThreshold，对比度阈值用于过滤区域中的弱特征。阈值越大，检测器产生的特征越少。；
# edgeThreshold ，用于过滤掉类似边缘特征的阈值。 请注意，其含义与contrastThreshold不同，即edgeThreshold越大，滤出的特征越少；
# sigma，高斯输入层级， 如果图像分辨率较低，则可能需要减少数值。

# sift = cv2.SIFT_create(nfeatures=128, nOctaveLayers=3, contrastThreshold=0.005, edgeThreshold=0.6, sigma=1.6)
sift = cv2.SIFT_create(nfeatures=1024, nOctaveLayers=3, contrastThreshold=0.0025, edgeThreshold=0.3, sigma=1.1)
# sift = cv2.AKAZE_create()
kps1, features1 = sift.detectAndCompute(img3, None)
kps2, features2 = sift.detectAndCompute(img4, None)

img3 = cv2.drawKeypoints(img3, kps1, img3, color=(255, 0, 255))  # 画出特征点，并显示为红色圆圈

img4 = cv2.drawKeypoints(img4, kps2, img4, color=(255, 0, 255))  # 画出特征点，并显示为红色圆圈

hmerge = np.hstack((img3, img4))  # 水平拼接

cv2.namedWindow("point", cv2.WINDOW_NORMAL)

cv2.imshow("point", hmerge)  # 拼接显示为gray

cv2.waitKey(0)

# 使用暴力匹配算法进行特征匹配，可以使用不同的距离度量方法（例如L2范数、汉明距离等）来计算特征之间的距离。由于算法的简单性，该方法在匹配小型数据集时效果较好，但在处理大型数据集时可能较慢。
# bf = cv2.BFMatcher_create(cv2.NORM_L1,crossCheck=True)
# bf = cv2.BFMatcher_create(cv2.NORM_L2,crossCheck=True)
# # 是立体匹配的特征匹配器，用于计算左右相机的深度图。
# bf = cv2.StereoMatcher()
# # 是描述符匹配的抽象基类，提供了特征匹配的接口，包括匹配方法、距离测量方法、k近邻等。
# bf = cv2.DescriptorMatcher()
# # 使用快速最近邻搜索算法进行特征匹配，例如k-d树和k-means聚类。该方法适用于处理大型数据集，但可能需要较长的训练时间。
bf = cv2.FlannBasedMatcher()

matches = bf.match(features1, features2)



#
# pt1 = np.float32([kps1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
# pt2 = np.float32([kps2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
# H, inliers,max_confidence = mlesac(pt1, pt2, max_iterations=1000, threshold=3)
#
#
# matches_mask = np.zeros(len(matches)).astype(np.uint8)
# matches_mask[inliers] = 1
# draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matches_mask, flags=2)
#
# img5 = cv2.drawMatches(img1, kps1, img2, kps2, matches, None, **draw_params)
#
# # img5 = cv2.drawMatches(img1, kps1, img2, kps2, matches, None, flags=2)
#
# cv2.namedWindow("BFmatch", cv2.WINDOW_NORMAL)
# cv2.imshow("BFmatch", img5)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




# 仅保留两张图片感兴趣区域的特征点
# 找到斜率相似的匹配点对
similar_matches = []
for match in matches:
    x1, y1 = kps1[match.queryIdx].pt
    x2, y2 = kps2[match.trainIdx].pt
    if x1>983 and x1<983+506 and y1>455 and y1<455+819 and x2>666 and x2 <666+578  and y2>397 and y2    <397+937:
    # if x1 in range(700, 983 + 506) and y1 in range(300, 455 + 819) and x2 in range(400, 666 + 578) and y2 in range(
    #         397, 397 + 937):
        similar_matches.append(match)
matches = []
matches = similar_matches
similar_matches = []

img5 = cv2.drawMatches(img1, kps1, img2, kps2, matches, None, flags=2)

cv2.namedWindow("BFmatch", cv2.WINDOW_NORMAL)
cv2.imshow("BFmatch", img5)
cv2.waitKey(0)
cv2.destroyAllWindows()

distances = []
slopes = []
max_distances = -10000
min_distances = 10000
max_slope = -10000
min_slope = 10000
for match in matches:

    x1, y1 = kps1[match.queryIdx].pt
    x2, y2 = kps2[match.trainIdx].pt
    slope = (y2 - y1) / (x2 - x1 + 1e-9)
    dis = math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
    if dis > max_distances:
        max_distances = dis
    if dis < min_distances:
        min_distances = dis
    distances.append(dis)
    if slope > max_slope:
        max_slope = slope
    if slope < min_slope:
        min_slope = slope
    slopes.append(slope)

distances_area = (max_distances - min_distances) * 0.1

max_slope = math.atan(max_slope)
min_slope = math.atan(min_slope)

# max_slope = math.radians(max_slope)
# min_slope = math.radians(min_slope)

max_slope = math.degrees(max_slope)
min_slope = math.degrees(min_slope)

slopes_area = (max_slope - min_slope) * 0.1
if slopes_area <= 0.4:
    slopes_area = 0.4

dist_kde = stats.gaussian_kde(distances)
slope_kde = stats.gaussian_kde(slopes)
# 计算距离和斜率概率密度函数的值
dist_values = dist_kde(distances)
slope_values = slope_kde(slopes)

max_dist_index = np.argmax(dist_values)
max_slope_index = np.argmax(slope_values)

max_dist = distances[max_dist_index]
max_slope_kde = slopes[max_slope_index]
max_slope_kde = math.atan(max_slope_kde)
max_slope_kde = math.degrees(max_slope_kde)

# 找到斜率相似的匹配点对
similar_matches = []
for i in range(len(matches)):
    slope_i = math.atan(slopes[i])
    slope_i = math.degrees(slope_i)
    if (abs(max_dist - distances[i]) < distances_area) and (abs(max_slope_kde - slope_i) < slopes_area):
        similar_matches.append(matches[i])

img5 = cv2.drawMatches(img1, kps1, img2, kps2, similar_matches, None, flags=2)

cv2.namedWindow("BFmatch", cv2.WINDOW_NORMAL)
cv2.imshow("BFmatch", img5)
cv2.waitKey(0)
cv2.destroyAllWindows()

distances = []
slopes = []
matches = []
matches = similar_matches
similar_matches = []

distances = []
slopes = []
max_distances = -10000
min_distances = 10000
max_slope = -10000
min_slope = 10000
for match in matches:

    x1, y1 = kps1[match.queryIdx].pt
    x2, y2 = kps2[match.trainIdx].pt
    slope = (y2 - y1) / (x2 - x1 + 1e-9)
    dis = math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
    if dis > max_distances:
        max_distances = dis
    if dis < min_distances:
        min_distances = dis
    distances.append(dis)
    if slope > max_slope:
        max_slope = slope
    if slope < min_slope:
        min_slope = slope
    slopes.append(slope)

distances_area = (max_distances - min_distances) * 0.1

max_slope = math.atan(max_slope)
min_slope = math.atan(min_slope)

max_slope = math.degrees(max_slope)
min_slope = math.degrees(min_slope)

slopes_area = (max_slope - min_slope) * 0.1
if slopes_area <= 0.4:
    slopes_area = 0.4

dist_kde = stats.gaussian_kde(distances)
slope_kde = stats.gaussian_kde(slopes)
# 计算距离和斜率概率密度函数的值
dist_values = dist_kde(distances)
slope_values = slope_kde(slopes)

max_dist_index = np.argmax(dist_values)
max_slope_index = np.argmax(slope_values)

max_dist = distances[max_dist_index]
max_slope_kde = slopes[max_slope_index]
max_slope_kde = math.atan(max_slope_kde)
max_slope_kde = math.degrees(max_slope_kde)

test_distance = []
test_slope = []
# 找到斜率相似的匹配点对
similar_matches = []
for i in range(len(matches)):
    slope_i = math.atan(slopes[i])
    slope_i = math.degrees(slope_i)
    if (abs(max_dist - distances[i]) < distances_area) and (abs(max_slope_kde - slope_i) < slopes_area):
        test_distance.append(distances[i])
        test_slope.append(slope_i)
        similar_matches.append(matches[i])

# img5 = cv2.drawMatches(img1, kps1, img2, kps2, similar_matches, None, flags=2)
#
# cv2.namedWindow("BFmatch", cv2.WINDOW_NORMAL)
# cv2.imshow("BFmatch", img5)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cnt = 0
# 选择最佳匹配点
good_matches = []
for m in similar_matches:
    cnt = cnt + 1
    # if m.distance < 0.90 * (len(features1[0]) ** 0.5):
    # if cnt < int(len(similar_matches)*0.2):
    # if cnt < int(len(similar_matches)*0.2) and cnt < 30:
    good_matches.append(m)

img5 = cv2.drawMatches(img1, kps1, img2, kps2, good_matches, None, flags=2)

cv2.namedWindow("BFmatch", cv2.WINDOW_NORMAL)
cv2.imshow("BFmatch", img5)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 提取匹配点的坐标
src_pts = np.float32([kps1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kps2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# 计算透视变换矩阵
M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 6.0)

img1 = cv2.imread("testv/v1/css-R-LM.jpg",cv2.IMREAD_UNCHANGED)
img2 = cv2.imread("testv/v1/css-R-LL.jpg",cv2.IMREAD_UNCHANGED)

# 对其中一张图片进行变换
result = cv2.warpPerspective(img1, M, (img1.shape[1] + img2.shape[1], img1.shape[0]+img2.shape[0]))

cv2.namedWindow("Panorama", cv2.WINDOW_NORMAL)
cv2.imshow('Panorama', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

img2 = cv2.copyMakeBorder(img2,0,result.shape[0]-img2.shape[0],0,result.shape[1]-img2.shape[1],cv2.BORDER_CONSTANT)


img2 = Image.fromarray(cv2.cvtColor(img2,cv2.COLOR_BGRA2RGBA))
result = Image.fromarray(cv2.cvtColor(result,cv2.COLOR_BGRA2RGBA))

# img2.paste(result, (0, 0))
out = Image.composite(result, img2, result)

out.save("luckylucky.png")
