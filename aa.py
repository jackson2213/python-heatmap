# Prolog - Auto Generated #
import os, uuid, matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot
import pandas
import cv2
import sys
import numpy as np
from PIL import Image
from pyheatmap.heatmap import HeatMap



def apply_heatmap(image,data):
    '''image是原图，data是坐标'''
    image1=cv2.imread(image)
    background = Image.new("RGB", (image1.shape[1], image1.shape[0]), color=0)
    # 开始绘制热度图
    hm = HeatMap(data)
    hit_img = hm.heatmap(base=background, r = 30) # background为背景图片，r是半径，默认为10
    hit_img = cv2.cvtColor(np.asarray(hit_img),cv2.COLOR_RGB2BGR)#Image格式转换成cv2格式
    overlay = image1.copy()
    alpha = 0.5 # 设置覆盖图片的透明度
    cv2.rectangle(overlay, (0, 0), (image1.shape[1], image1.shape[0]), (255, 0, 0), -1) # 设置蓝色为热度图基本色蓝色
    image2 = cv2.addWeighted(overlay, alpha, image1, 1-alpha, 0) # 将背景热度图覆盖到原图
    image3 = cv2.addWeighted(hit_img, alpha, image2, 1-alpha, 0) # 将热度图覆盖到原图
    cv2.imshow('ru',image3)
    cv2.imwrite('test.jpg',image3)
    cv2.waitKey(0)
    return image3



dataset = pandas.read_csv('input_df_b31ec4e4-69ad-479f-9ea0-058ea839065b.csv')
data=(dataset.loc[:,['x','y','区域1总人次']].values/2).tolist()
data= [list(map(int, i)) for i  in data]
# Epilog - Auto Generated #


if __name__ == '__main__':
    apply_heatmap('11.jpg',data)