# Prolog - Auto Generated #
import os, uuid, matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot
import pandas
import matplotlib.image as mpimg
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from PIL import ImageDraw2

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

def get_max_size(data):
    max_w = 0
    max_h = 0
    for hit in data:
        w = hit[0]
        h = hit[1]
        if w > max_w:
            max_w = w
        if h > max_h:
            max_h = h

    return max_w + 1, max_h + 1

def mk_colors(n=240):
    u"""生成色盘
    @see http://oldj.net/article/heat-map-colors/

    TODO: 根据 http://oldj.net/article/hsl-to-rgb/ 将 HSL 转为 RGBA
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
    def __init__(self,
                 data,
                 base=None,
                 width=0,
                 height=0
                 ):
        u""""""

        assert type(data) in (list, tuple)
        assert base is None or os.path.isfile(base)
        assert width >= 0 and height >= 0

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
        self.count = count
        self.base = base
        self.width = width
        self.height = height
        self.save_as = None

        if not self.base and (self.width == 0 or self.height == 0):
            w, h = get_max_size(data)
            self.width = self.width or w
            self.height = self.height or h

    def __mk_img(self, base=None):
        u"""生成临时图片"""

        base = base or self.base
        self.__im0 = None

        if base:
            str_type = (str,)
            self.__im0 = Image.open(base) if type(base) in str_type else base
            self.width, self.height = self.__im0.size

        self.__im = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))



    def __heat(self, heat_data, x, y, n, template):
        u""""""

        l = len(heat_data)
        width = self.width
        p = width * y + x

        for ip, iv in template:
            p2 = p + ip
            if 0 <= p2 < l:
                heat_data[p2] += iv * n

    def __paint_heat(self, heat_data, colors):
        u""""""

        import re

        im = self.__im
        rr = re.compile(", (\d+)%\)")
        dr = ImageDraw2.ImageDraw.Draw(im)
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
                    im.putpixel((x, y), (0, 0, 255, al))
                else:
                    dr.point((x, y), fill=color)

    def __add_base(self):
        if not self.__im0:
            return

        self.__im0.paste(self.__im, mask=self.__im)
        self.__im = self.__im0



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



dataset = pandas.read_csv('input_df_b31ec4e4-69ad-479f-9ea0-058ea839065b.csv')
data=(dataset.loc[:,['x','y','区域1总人次']].values/2).tolist()
data= [list(map(int, i)) for i  in data]
# Epilog - Auto Generated #


if __name__ == '__main__':
    apply_heatmap('11.jpg',data)