import cv2
import numpy as np
import pandas as pd
import os


def get_coor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []  # 存储所有对象的边界框的列表

    for contour in contours:
        a = sorted(contour[:, 0], key=lambda x: x[0])
        x_min = a[0][0]
        x_max = a[-1][0]
        b = sorted(contour[:, 0], key=lambda x: x[1])
        y_min = b[0][1]
        y_max = b[-1][1]
        bounding_boxes.append((x_min, y_min, x_max, y_max))

    return bounding_boxes


def save_xml(src_xml_dir, img_name, h, w, bounding_boxes):
    xml_file = open((src_xml_dir + '/' + img_name + '.xml'), 'w')
    xml_file.write('<annotation>\n')
    xml_file.write('    <folder>VOC2012</folder>\n')

    xml_file.write('    <filename>' + str(img_name) + '.jpg' + '</filename>\n')

    xml_file.write('    <size>\n')
    xml_file.write('        <width>' + str(w) + '</width>\n')
    xml_file.write('        <height>' + str(h) + '</height>\n')
    xml_file.write('        <depth>3</depth>\n')
    xml_file.write('    </size>\n')

    for i, (x_min, y_min, x_max, y_max) in enumerate(bounding_boxes):
        xml_file.write('    <object>\n')
        xml_file.write('        <name>' + 'ma' + '</name>\n')
        xml_file.write('        <pose>Unspecified</pose>\n')
        xml_file.write('        <truncated>0</truncated>\n')
        xml_file.write('        <difficult>0</difficult>\n')
        xml_file.write('        <bndbox>\n')
        xml_file.write('            <xmin>' + str(x_min) + '</xmin>\n')
        xml_file.write('            <ymin>' + str(y_min) + '</ymin>\n')
        xml_file.write('            <xmax>' + str(x_max) + '</xmax>\n')
        xml_file.write('            <ymax>' + str(y_max) + '</ymax>\n')
        xml_file.write('        </bndbox>\n')
        xml_file.write('    </object>\n')


    xml_file.write('</annotation>')
    xml_file.close()

file_dir = 'MA'
save_xml_dir = 'MA_out'
for name in os.listdir(file_dir):
    print(name)
    img_path = os.path.join(file_dir, name)
    img = cv2.imread(img_path)
    # img = cv2.imdecode(np.fromfile(img_path), dtype=np.uint8), -1)
    h, w = img.shape[:-1]
    bounding_boxes = get_coor(img)
    img_name = name.split('.')[0]
    save_xml(save_xml_dir, img_name, h, w, bounding_boxes)