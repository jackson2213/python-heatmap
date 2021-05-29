# Prolog - Auto Generated #
import os, uuid, matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot
import pandas
import matplotlib.image as mpimg
import sys
import numpy as np
from PIL import Image
from pyheatmap.heatmap import HeatMap
import matplotlib.pyplot as plt


def apply_heatmap(image,data):
    '''image是原图，data是坐标'''
    image1=mpimg.imread(image)
    background = Image.new("RGB", (image1.shape[1], image1.shape[0]), color=0)
    # 开始绘制热度图
    hm = HeatMap(data)
    hit_img = hm.heatmap(base=background, r=30)  # background为背景图片，r是半径，默认为10
    alpha = 0.5 # 设置覆盖图片的透明度
    overlay = np.full(image1.shape, (0, 0, 255)) # 设置蓝色为热度图基本色蓝色
    image2=overlay*alpha + image1*(1-alpha) # 将背景热度图覆盖到原图
    image3= np.asarray(hit_img)*alpha + image2* (1-alpha) # 将热度图覆盖到原图

    plt.imshow(np.asarray(image3,dtype=int))

    plt.show()

    # cv2.imshow('ru',image3)
    # cv2.imwrite('test.jpg',image3)
    # cv2.waitKey(0)
    # return image3



dataset = pandas.read_csv('input_df_b31ec4e4-69ad-479f-9ea0-058ea839065b.csv')
data=(dataset.loc[:,['x','y','区域1总人次']].values/2).tolist()
data= [list(map(int, i)) for i  in data]
# Epilog - Auto Generated #


if __name__ == '__main__':
    apply_heatmap('11.jpg',data)