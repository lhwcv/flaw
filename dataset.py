from tensorpack import RNGDataFlow
import os
import  cv2
import  numpy as np
from utils import get_data_dict
from PIL import Image
import random
SRC_IMG_W=2560
SRC_IMG_H=1920

## 第一阶段缺陷检测要求输出有缺陷的概率
## 尝试将有缺陷的布匹的bndbox填成前景, 转化为分割问题
class FlawDetect_Stage1_DataSet(RNGDataFlow):
    def __init__(self,datadir,shuffle=True):
        self.reset_state()
        assert  os.path.isdir(datadir)
        self.shuffle=shuffle
        print('preloading data....')
        self._preload(datadir)
        print('data preloaded !')
    def _preload(self,datadir):
        data_dict = get_data_dict(datadir)
        self.data_dict={}
        for label in data_dict.keys():
            data = data_dict[label]
            for fname in data['filelist']:
                if label=='正常':
                    self.data_dict[fname]={'is_normal':True}
                else:
                    self.data_dict[fname]={'is_normal':False, 'bboxes':data['anno'][fname]['bboxes']}
        self.image_files = list(self.data_dict.keys())
        print('total images: ', len(self.image_files))
    def size(self):
        return len(self.image_files)
    def get_data(self):
        idxs = np.arange(self.size())
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            #yield [self.data[k], self.label[k]]
            name = self.image_files[k]
            img = cv2.imread(name, cv2.IMREAD_COLOR)
            assert img is not None
            assert img.shape[:2] == (SRC_IMG_H, SRC_IMG_W)
            label = np.zeros((SRC_IMG_H, SRC_IMG_W, 1), dtype='float32')
            if not self.data_dict[name]['is_normal']:
                for b in self.data_dict[name]['bboxes']:
                    b = b['bb']
                    label[b[1]:b[3], b[0]:b[2]] = 1.0
            yield [img,label]


class FlawDetect_Stage1_Classify_DataSet(RNGDataFlow):
    def __init__(self,normal_path, abnormal_path,shuffle=True):
        self.reset_state()
        assert  os.path.isdir(normal_path)
        assert  os.path.isdir(abnormal_path)
        self.shuffle=shuffle
        self.normal_imgs = [os.path.join(normal_path,f) for f in os.listdir(normal_path) if 'jpg' in f]
        self.abnormal_imgs = [os.path.join(abnormal_path,f) for f in os.listdir(abnormal_path) if 'jpg' in f]


    def size(self):
        return len(self.normal_imgs)+len(self.abnormal_imgs)
    def get_data(self):
        while(True):
            label=0  #normal
            name = self.rng.choice(self.normal_imgs)
            ## ensure normal/abnormal= 2/1
            rnd = self.rng.randint(low=0, high=3)
            if rnd==0:
                label=1
                name = self.rng.choice(self.abnormal_imgs)
            img = cv2.imread(name, cv2.IMREAD_COLOR)
            assert img is not None
            yield [img,label]



if __name__=='__main__':
    DATA_PATH = '/home/lhw/Dataset/ali_flaw/xuelang_round1_train_part1_20180628'
    normal_path = '/home/lhw/Dataset/ali_flaw/clc_data/normal'
    abnormal_path = '/home/lhw/Dataset/ali_flaw/clc_data/abnormal'
    df =FlawDetect_Stage1_Classify_DataSet(normal_path,abnormal_path)
    for k in df.get_data():
        cv2.namedWindow('img',0)
        cv2.imshow("img", k[0])
        print(k[1])
        cv2.waitKey(0)
    # df = FlawDetect_Stage1_DataSet(DATA_PATH)
    # for k in df.get_data():
    #     cv2.namedWindow('img',0)
    #     cv2.namedWindow('label',0)
    #     cv2.imshow("img", k[0])
    #     cv2.imshow("label", k[1] * 255)
    #     cv2.waitKey(0)





