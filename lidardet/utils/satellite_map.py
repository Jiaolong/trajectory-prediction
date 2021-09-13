'''
Download web map by cooridinates

'''

import io
import cv2
import math
import numpy as np
import PIL.Image as pil
from pathlib import Path
import urllib.request as ur
from threading import Thread, Lock
from math import floor, pi, log, tan, atan, exp

CACHE_DIR = Path.home() / '.cache' / 'sirius' / 'satellite_map'
MAP_URLS = {
    "google2": "http://mt2.google.cn/vt/lyrs={style}&hl=zh-CN&gl=CN&src=app&x={x}&y={y}&z={z}",
    "google": "http://mt3.google.cn/vt/lyrs={style}&scale=1&hl=zh-CN&x={x}&y={y}&z={z}",
    "amap": "http://wprd02.is.autonavi.com/appmaptile?style={style}&x={x}&y={y}&z={z}",
    "tencent_s": "http://p3.map.gtimg.com/sateTiles/{z}/{fx}/{fy}/{x}_{y}.jpg",
    "tencent_m": "http://rt0.map.gtimg.com/tile?z={z}&x={x}&y={y}&styleid=3"  }

COUNT=0
mutex=Lock()


#-----------------GCJ02到WGS84的纠偏与互转---------------------------
def transformLat(x, y):
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(y * math.pi) + 40.0 * math.sin(y / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(y / 12.0 * math.pi) + 320 * math.sin(y * math.pi / 30.0)) * 2.0 / 3.0
    return ret

def transformLon(x, y):
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(x * math.pi) + 40.0 * math.sin(x / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(x / 12.0 * math.pi) + 300.0 * math.sin(x / 30.0 * math.pi)) * 2.0 / 3.0
    return ret


def delta(lat, lon):
    ''' 
    Krasovsky 1940
    //
    // a = 6378245.0, 1/f = 298.3
    // b = a * (1 - f)
    // ee = (a^2 - b^2) / a^2;
    '''
    a = 6378245.0   #  a: 卫星椭球坐标投影到平面地图坐标系的投影因子。
    ee = 0.00669342162296594323   #  ee: 椭球的偏心率。
    dLat = transformLat(lon - 105.0, lat - 35.0)
    dLon = transformLon(lon - 105.0, lat - 35.0)
    radLat = lat / 180.0 * math.pi
    magic = math.sin(radLat)
    magic = 1 - ee * magic * magic
    sqrtMagic = math.sqrt(magic)
    dLat = (dLat * 180.0) / ((a * (1 - ee)) / (magic * sqrtMagic) * math.pi)
    dLon = (dLon * 180.0) / (a / sqrtMagic * math.cos(radLat) * math.pi)
    return {'lat': dLat, 'lon': dLon}

def outOfChina(lat, lon):
    if (lon < 72.004 or lon > 137.8347):
        return True
    if (lat < 0.8293 or lat > 55.8271):
        return True
    return False

def gcj_to_wgs(gcjLon,gcjLat):
    if outOfChina(gcjLat, gcjLon):
        return (gcjLon, gcjLat)
    d = delta(gcjLat, gcjLon)
    return (gcjLon - d["lon"],gcjLat - d["lat"])

def wgs_to_gcj(wgsLon,wgsLat):
    if outOfChina(wgsLat, wgsLon):
        return wgsLon, wgsLat
    d = delta(wgsLat, wgsLon);
    return wgsLon + d["lon"], wgsLat + d["lat"]

#--------------------------------------------------------------

#------------------wgs84与web墨卡托互转-------------------------

# WGS-84经纬度转Web墨卡托
def wgs_to_macator(x, y):
    y = 85.0511287798 if y > 85.0511287798 else y
    y = -85.0511287798 if y < -85.0511287798 else y

    x2 = x * 20037508.34 / 180
    y2 = log(tan((90 + y) * pi / 360)) / (pi / 180)
    y2 = y2 * 20037508.34 / 180
    return x2, y2

# Web墨卡托转经纬度
def mecator_to_wgs(x, y):
    x2 = x / 20037508.34 * 180
    y2 = y / 20037508.34 * 180
    y2 = 180 / pi * (2 * atan(exp(y2 * pi / 180)) - pi / 2)
    return x2, y2

#-------------------------------------------------------------

#---------------------------------------------------------
'''
东经为正，西经为负。北纬为正，南纬为负
j经度 w纬度 z缩放比例[0-22] ,对于卫星图并不能取到最大，测试值是20最大，再大会返回404.
山区卫星图可取的z更小，不同地图来源设置不同。
'''
# 根据WGS-84 的经纬度获取谷歌地图中的瓦片坐标
def wgs84_to_tile(j, w, z):
    '''
    Get google-style tile cooridinate from geographical coordinate
    j : Longittude
    w : Latitude
    z : zoom
    '''
    isnum = lambda x: isinstance(x, int) or isinstance(x, float)
    if not(isnum(j) and isnum(w)):
        raise TypeError("j and w must be int or float!")

    if not isinstance(z, int) or z < 0 or z > 22:
        raise TypeError("z must be int and between 0 to 22.")

    if j < 0:
        j = 180 + j
    else:
        j += 180
    j /= 360  # make j to (0,1)

    w = 85.0511287798 if w > 85.0511287798 else w
    w = -85.0511287798 if w < -85.0511287798 else w
    w = log(tan((90 + w) * pi / 360)) / (pi / 180)
    w /= 180  # make w to (-1,1)
    w = 1 - (w + 1) / 2  # make w to (0,1) and left top is 0-point

    num = 2**z
    x = floor(j * num)
    y = floor(w * num)
    return x, y

def tileframe_to_mecatorframe(zb):
    # 根据瓦片四角坐标，获得该区域四个角的web墨卡托投影坐标
    inx, iny =zb["LT"]   #left top
    inx2,iny2=zb["RB"] #right bottom
    length = 20037508.3427892
    sum = 2**zb["zoom"]
    LTx = inx / sum * length * 2 - length
    LTy = -(iny / sum * length * 2) + length

    RBx = (inx2 + 1) / sum * length * 2 - length
    RBy = -((iny2 + 1) / sum * length * 2) + length

    # LT=left top,RB=right buttom
    # 返回四个角的投影坐标
    res = {'LT': (LTx, LTy), 'RB': (RBx, RBy),
           'LB': (LTx, RBy), 'RT': (RBx, LTy)}
    return res

def tileframe_to_pixframe(zb):
    # 瓦片坐标转化为最终图片的四个角像素的坐标
    out={}
    width=(zb["RT"][0]-zb["LT"][0]+1)*256
    height=(zb["LB"][1]-zb["LT"][1]+1)*256
    out["LT"]=(0,0)
    out["RT"]=(width,0)
    out["LB"]=(0,-height)
    out["RB"]=(width,-height)
    return out

#-----------------------------------------------------------

class Downloader(Thread):
    # multiple threads downloader
    def __init__(self,index,count,urls,datas,update):
        # index 表示第几个线程，count 表示线程的总数，urls 代表需要下载url列表，datas代表要返回的数据列表。
        # update 表示每下载一个成功就进行的回调函数。
        super().__init__()
        self.urls=urls
        self.datas=datas
        self.index=index
        self.count=count
        self.update=update

    def download(self,url):
        #HEADERS = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/29.0.1547.76 Safari/537.36'}
        header = ur.Request(url)#,headers=HEADERS)
        err=0
        while(err<30):
            try:
                data = ur.urlopen(header).read()
            except:
                err+=1
            else:
                return data
        #raise Exception("Bad network link.")
        print("Download satellite map failed! Bad network link.")
        return None

    def run(self):
        for i,url in enumerate(self.urls):
            if i%self.count != self.index:
                continue
            self.datas[i]=self.download(url)
            if mutex.acquire():
                self.update()
                mutex.release()

def geturl(source, x, y, z, style):
    '''
    Get the picture's url for download.
    style:
        m for map
        s for satellite
    source:
        google or amap or tencent
    x y:
        google-style tile coordinate system
    z:
        zoom 
    '''
    if source == 'google':
        furl = MAP_URLS["google"].format(x=x, y=y, z=z, style=style)
    elif source == 'amap':
        # for amap 6 is satellite and 7 is map.
        style = 6 if style == 's' else 7
        furl = MAP_URLS["amap"].format(x=x, y=y, z=z, style=style)
    elif source == 'tencent':
        y = 2**z - 1 - y
        if style == 's':
            furl = MAP_URLS["tencent_s"].format(
                x=x, y=y, z=z, fx=floor(x / 16), fy=floor(y / 16))
        else:
            furl = MAP_URLS["tencent_m"].format(x=x, y=y, z=z)
    else:
        raise Exception("Unknown Map Source ! ")

    return furl

def downpics(urls,multi=10):

    def makeupdate(s):
        def up():
            global COUNT
            COUNT+=1
            print("\b"*45,end='')
            print("DownLoading ... [{0}/{1}]".format(COUNT,s),end='')
        return up

    url_len=len(urls)
    datas=[None] * url_len
    if multi <1 or multi >20 or not isinstance(multi,int):
        raise Exception("multi of Downloader shuold be int and between 1 to 20.")
    tasks=[Downloader(i,multi,urls,datas,makeupdate(url_len)) for i in range(multi)]
    for i in tasks:
        i.start()
    for i in tasks:
        i.join()

    return datas


def get_satellite_image(lat_min, lat_max, lon_min, lon_max, crop=True):
    img, tile = get_map(lon_min, lat_max, lon_max, lat_min,
                17, source='google', style='s')
    if img is not None and crop:
        img = crop_map(img, tile, [lat_max, lon_min, lat_min, lon_max])

    return img, tile

def get_map(x1, y1, x2, y2, zoom, source='google', style='s'):
    '''
    依次输入左上角的经度、纬度，右下角的经度、纬度，缩放级别，地图源，输出文件，影像类型（默认为卫星图）
    获取区域内的瓦片并自动拼合图像。返回四个角的瓦片坐标
    '''

    pos1x, pos1y = wgs84_to_tile(x1, y1, zoom)
    pos2x, pos2y = wgs84_to_tile(x2, y2, zoom)
    lenx = pos2x - pos1x + 1
    leny = pos2y - pos1y + 1
    tile =  {"LT":(pos1x,pos1y),"RT":(pos2x,pos1y),"LB":(pos1x,pos2y),"RB":(pos2x,pos2y),"zoom":zoom}
    print("Total number of tiles: {x} X {y}".format(x=lenx, y=leny))
    
    img_name = '{}-{}-{}-{}-{}.jpg'.format(pos1x, pos1y, pos2x, pos2y, zoom)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    save_path = CACHE_DIR / img_name
    if save_path.exists(): 
        print('Load satellite image from cache')
        img = cv2.imread(str(save_path))
        return img, tile

    urls = [geturl(source, i, j, zoom, style) for j in range(pos1y, pos1y + leny) for i in range(pos1x, pos1x + lenx)]
    #print(urls)
    datas = downpics(urls)

    outpic = pil.new('RGB', (lenx * 256, leny * 256))
    for i, data in enumerate(datas):
        if data is None:
            return None, tile
        picio = io.BytesIO(data)
        small_pic = pil.open(picio)

        y, x = i // lenx, i % lenx
        outpic.paste(small_pic, (x * 256, y * 256))

    img = np.array(outpic)[...,::-1].astype('uint8') # RGB -> BGR
    cv2.imwrite(str(save_path), img)
    print('Satellite map has been downloaded and cached at {}!'.format(str(save_path)))
    return img, tile

def crop_map(map_img, tile, crop_range):
    zoom = tile['zoom']

    width = (tile["RT"][0] - tile["LT"][0] + 1) * 256
    height = (tile["LB"][1] - tile["LT"][1] + 1) * 256
    tile_m = tileframe_to_mecatorframe(tile)
    mx0 = tile_m['LT'][0]
    my0 = tile_m['LT'][1]
    mx1 = tile_m['RB'][0]
    my1 = tile_m['RB'][1]
    res = (mx1 - mx0) / width

    lat1, lon1, lat2, lon2 = crop_range

    mx1, my1 = wgs_to_macator(lon1, lat1)
    # tile coordinate to pixel coordinate
    px1 = int((mx1 - mx0 + 1) / res)
    py1 = int((my0 - my1 - 1) / res)

    mx2, my2 = wgs_to_macator(lon2, lat2)
    # tile coordinate to pixel coordinate
    px2 = int((mx2 - mx0 + 1) / res)
    py2 = int((my0 - my2 - 1) / res)

    return map_img[py1:py2,px1:px2:,:]
