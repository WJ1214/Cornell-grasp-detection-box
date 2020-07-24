import os
import sys
import cv2
import math
import numpy as np
from tqdm import tqdm
from decimal import Decimal
from PIL import Image
import random
import cv2
import math
import random


# 将原本的cornell原数据转换为项目能够使用的yolo格式数据
# cornell数据格式每行为一个点，每4行为一个抓取框
# 需要的yolo格式数据： c,x,y,w,h,a(c为类别，默认全部为0)
# x,y,w,h全部需要将其归一化为yolo格式，范围为(0, 1), a范围为(-90, 90)
# 每张图片对应一个命名相同的txt文件


def convert(src_path, target_path):
    img_path = "{}/image".format(src_path)
    label_path = "{}/pos_label".format(src_path)
    img_name = os.listdir(img_path)
    for i, img in enumerate(img_name):
        img = img.rstrip("r.png")
        img_name[i] = img
    for name in img_name:
        label_name = "{}cpos.txt".format(name)
        labelpath = "{}/{}".format(label_path, label_name)
        # 打开要写的label文件
        with open(os.path.join(target_path, name + "r.txt"), 'w') as f:
            image_box = []
            # 打开要读的label文件
            with open(labelpath, 'r', encoding='utf-8-sig') as fl:
                class_label = 0
                image = name + "r.png"
                height, width, _ = cv2.imread("{}/{}".format(img_path, image)).shape
                for lines in fl:
                    image_box.append(list(map(float, lines.split())))
                image_box = np.array(image_box).astype(np.int32)
                num = image_box.shape[0]
                num = int(num / 4)
                image_box = image_box.reshape((num, 8))

                # ((cx, cy), (w, h), theta) = cv2.minAreaRect(image_box)
                # a = theta
                # if a >  0.5*math.pi:
                #     a = math.pi - a
                # if a < -0.5*math.pi:
                #     a = math.pi + a
                xywh = points2xywh(image_box)
                angles = []
                boxes = numpy_box2list(image_box)
                for i, elm in enumerate(boxes):
                    angle = calculate_angle(elm)
                    angles.append([angle])
                angles = np.array(angles)
                xywha = np.hstack((xywh, angles))
                for box in xywha:
                    cx = box[0]
                    cy = box[1]
                    w = box[2]
                    h = box[3]
                    a = box[4]
                    a = a / 180 * math.pi

                    x = Decimal(cx / width).quantize(Decimal('0.000000'))
                    y = Decimal(cy / height).quantize(Decimal('0.000000'))
                    w = Decimal(w / width).quantize(Decimal('0.000000'))
                    h = Decimal(h / height).quantize(Decimal('0.000000'))
                    a = Decimal(a).quantize(Decimal('0.000000'))
                    f.write(str(class_label) + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + ' ' + str(
                        a) + '\n')


def check_angle(src_path):
    file_names = os.listdir(src_path)
    for name in tqdm(file_names):
        with open(os.path.join(src_path, name)) as f:
            data = []
            for line in f:
                data.append(list(map(float, line.split())))
            data = np.array(data)
            for i, angle in enumerate(data[:, 5]):
                if angle > math.pi / 2 or angle < -math.pi / 2:
                    print(name)
                    print(i)


def numpy_box2list(bbox):
    # 输入一张图片的相应numpy格式抓取框
    res = []
    for elm in bbox:
        box = []
        vex1 = list(elm[:2])
        vex2 = list(elm[2:4])
        vex3 = list(elm[4:6])
        vex4 = list(elm[6:8])
        box.append(vex1)
        box.append(vex2)
        box.append(vex3)
        box.append(vex4)
        res.append(box)
    return res


def calculate_x_angle(bbox):
    # 输入一个list格式的抓取框
    vex1 = bbox[0]
    vex2 = bbox[1]
    arr_a = np.array([(vex2[0] - vex1[0]), (vex2[1] - vex1[1])])  # 向量a
    axis_x = np.array([1, 0])  # 向量b
    cos_value = (float(arr_a.dot(axis_x)) / (np.sqrt(arr_a.dot(arr_a)) * np.sqrt(axis_x.dot(axis_x))))
    return np.arccos(cos_value) * (180 / np.pi)


def calculate_y_angle(bbox):
    # 输入一个list格式的抓取框
    vex1 = bbox[0]
    vex2 = bbox[1]
    arr_a = np.array([(vex2[0] - vex1[0]), (vex2[1] - vex1[1])])  # 向量a
    axis_y = np.array([0, 1])  # 向量b
    cos_value = (float(arr_a.dot(axis_y)) / (np.sqrt(arr_a.dot(arr_a)) * np.sqrt(axis_y.dot(axis_y))))
    return np.arccos(cos_value) * (180 / np.pi)


def calculate_angle(bbox):
    # 输入一个list类型的bbox,返回其应当顺时针旋转的角度
    # 将角度范围限定为（-90. 90），抓取夹水平收缩时为0度，顺时针旋转为-angle，逆时针旋转为angle
    # 抓取夹收缩方向为h
    # 抓取夹宽度为w
    x_angle = calculate_x_angle(bbox)
    y_angle = calculate_y_angle(bbox)
    if x_angle == 90:
        return 90
    if y_angle == 90:
        return 0
    if x_angle < 90 and y_angle < 90:
        return -x_angle
    if x_angle > 90 and y_angle > 90:
        return x_angle - 180
    if x_angle < 90 and y_angle > 90:
        return x_angle
    if x_angle > 90 and y_angle < 90:
        return 180 - x_angle


def points2xywh(bbox):
    # 输入一个numpy类型的bbox(4个点的坐标),返回其(x, y, w, h)，角度数据在数据集中已经给出
    res = []
    length = bbox.shape[0]

    for i in range(length):
        res_box = []
        box = bbox[i]
        x = (box[0] + box[4]) / 2
        y = (box[1] + box[5]) / 2
        # 抓取夹收缩方向为w
        # 抓取夹宽度为h
        w = math.sqrt((box[0] - box[2]) ** 2 + (box[1] - box[3]) ** 2)
        h = math.sqrt((box[2] - box[4]) ** 2 + (box[3] - box[5]) ** 2)
        res_box.append(x)
        res_box.append(y)
        res_box.append(w)
        res_box.append(h)
        res.append(res_box)
    res = np.array(res)
    return res



if __name__ == '__main__':
    convert("/home/datas/Cornell", "/home/wj/Cornell_yolo/labels")