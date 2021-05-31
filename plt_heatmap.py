# Prolog - Auto Generated #
# -*- coding: utf-8 -*-
import pandas
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import re

ONE_THIRD = 1.0/3.0
ONE_SIXTH = 1.0/6.0
TWO_THIRD = 2.0/3.0
def _v(m1, m2, hue):
    hue = hue % 1.0
    if hue < ONE_SIXTH:
        return m1 + (m2-m1)*hue*6.0
    if hue < 0.5:
        return m2
    if hue < TWO_THIRD:
        return m1 + (m2-m1)*(TWO_THIRD-hue)*6.0
    return m1
def hls_to_rgb(color):
    m = re.match(
        r"hsl\(\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)%\s*,\s*(\d+\.?\d*)%\s*\)$", color
    )
    h, l, s=float(m.group(1)) / 360.0,float(m.group(3)) / 100.0, float(m.group(2)) / 100.0,
    if s == 0.0:
        return l, l, l
    if l <= 0.5:
        m2 = l * (1.0+s)
    else:
        m2 = l+s-(l*s)
    m1 = 2.0*l - m2
    rgb= (_v(m1, m2, h+ONE_THIRD), _v(m1, m2, h), _v(m1, m2, h-ONE_THIRD))
    return (
        int(rgb[0] * 255 + 0.5),
        int(rgb[1] * 255 + 0.5),
        int(rgb[2] * 255 + 0.5),
        255
    )



def mk_circle(r, w):
    u"""根据半径r以及图片宽度 w ，产生一个圆的list
    @see http://oldj.net/article/bresenham-algorithm/
    """
    __tmp = {}
    def c8(ix, iy, v=1):
        # 8对称性
        ps = ( (ix, iy),(-ix, iy),(ix, -iy),(-ix, -iy), (iy, ix),(-iy, ix),(iy, -ix),(-iy, -ix),)
        for x2, y2 in ps:
            p = w * y2 + x2
            __tmp.setdefault(p, v)
            # __clist.add((p, v))

    # 中点圆画法
    x = 0
    y = r
    d = 3 - (r << 1)
    while x <= y:
        for _y in range(x, y + 1):
            c8(x, _y, y + 1 - _y)
        if d < 0:
            d += (x << 2) + 6
        else:
            d += ((x - y) << 2) + 10
            y -= 1
        x += 1
    return __tmp.items()





def mk_colors(n=240):
    u"""生成色盘
    """
    colors = []
    n1 = int(n * 0.4)
    n2 = n - n1

    for i in range(n1):
        color = "hsl(240, 100%%, %d%%)" % (100 * (n1 - i / 2) / n1)
        # color = 255 * i / n1
        colors.append(color)
    for i in range(n2):
        color = "hsl(%.0f, 100%%, 50%%)" % (240 * (1.0 - float(i) / n2))
        colors.append(color)
    return colors

class HeatMap(object):
    def __init__(self,data):
        assert type(data) in (list, tuple,np.ndarray)
        count = 0
        data2 = []
        for hit in data:
            if len(hit) == 3:
                x, y, n = hit
            elif len(hit) == 2:
                x, y, n = hit[0], hit[1], 1
            else:
                raise Exception(u"length of hit is invalid!")

            data2.append((x, y, n))
            count += n

        self.data = data2
        self.base = None
        self.width = 0
        self.height = 0

        if not self.base and (self.width == 0 or self.height == 0):
            w, h = max([i[0] for i in data]) + 1, max([i[1] for i in data]) + 1
            self.width = self.width or w
            self.height = self.height or h

    def __mk_img(self, base=None):
        u"""生成临时图片"""
        self.__im0= base
        self.width, self.height = base.shape[1], base.shape[0]  #self.__im0.shape  (160, 120)
        self.__im = np.full((self.height,self.width, 4), (0, 0, 0, 0))


    def __heat(self, heat_data, x, y, n, template):
        l = len(heat_data)
        width = self.width
        p = width * y + x
        for ip, iv in template:
            p2 = p + ip
            if 0 <= p2 < l:
                heat_data[p2] += iv * n

    def __paint_heat(self, heat_data, colors):
        import re
        im = self.__im
        rr = re.compile(", (\d+)%\)")
        width = self.width
        height = self.height
        max_v = max(heat_data)
        if max_v <= 0:
            # 空图片
            return

        r = 240.0 / max_v
        heat_data2 = [int(i * r) - 1 for i in heat_data]

        size = width * height

        for p in range(size):
            v = heat_data2[p]
            if v > 0:
                x, y = p % width, p // width
                color = colors[v]
                alpha = int(rr.findall(color)[0])
                if alpha > 50:
                    al = 255 - 255 * (alpha - 50) // 50
                    im[y][x]=(0, 0, 255, al)
                else:
                    im[y][x]=hls_to_rgb(color)

    def __add_base(self):
        front = np.asarray(self.__im)
        result = np.empty(front.shape, dtype='float')
        alpha = np.index_exp[:, :, 3:]
        rgb = np.index_exp[:, :, :3]
        falpha = front[alpha] / 255.0
        result[rgb] = (front[rgb] * falpha + self.__im0 * (1 - falpha))
        np.clip(result, 0, 255)
        result = result.astype('uint8')
        self.__im = result[rgb]


    def heatmap(self, base=None, data=None, r=10):
        u"""绘制热图"""
        self.__mk_img(base)
        circle = mk_circle(r, self.width)
        heat_data = [0] * self.width * self.height
        data = data or self.data

        for hit in data:
            x, y, n = hit
            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                continue
            self.__heat(heat_data, x, y, n, circle)

        self.__paint_heat(heat_data, mk_colors())
        self.__add_base()
        return self.__im



def apply_heatmap(image,data):
    '''image是原图，data是坐标'''
    image1=mpimg.imread(image)
    background = np.full(image1.shape, (0, 0, 0)) #生成黑色背景图
    # 开始绘制热度图
    hm = HeatMap(data)
    hit_img = hm.heatmap(base=background, r=30)  # background为背景图片，r是半径，默认为10
    alpha = 0.3 # 设置覆盖图片的透明度
    overlay = np.full(image1.shape, (0, 0, 255)) # 设置蓝色为热度图基本色蓝色
    image2=overlay*alpha + image1*(1-alpha) # 将背景热度图覆盖到原图
    image3= np.asarray(hit_img)*alpha + image2* (1-alpha) # 将热度图覆盖到原图

    plt.imshow(np.asarray(image3,dtype=int))

    plt.show()



dataset = pandas.read_csv('input_df_b31ec4e4-69ad-479f-9ea0-058ea839065b.csv')
data=(dataset.loc[:,['x','y','people']].values/2).tolist()
data=np.asarray(data,dtype=int)
# Epilog - Auto Generated #



apply_heatmap('D:\\pycharm_project\\python-heatmap\\img.jpg',data)