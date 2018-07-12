from tensorpack import *
from src.dataset import FlawDetect_Stage1_DataSet
import cv2
from src.unet import Unet
from src.utils import get_data_dict
import numpy as np
import tensorflow.contrib.slim as slim


DATA_PATH='/home/lhw/Dataset/ali_flaw/20180705'

def get_data(datadir):
    ds = FlawDetect_Stage1_DataSet(datadir)
    shape_aug = [
        imgaug.Resize((512,512)),

    ]
    ds = AugmentImageComponents(ds, shape_aug,(0,1), copy=False)
    ds = BatchDataByShape(ds, 16, idx=0)
    ds = PrefetchDataZMQ(ds, 2)
    return  ds

def view_data(datadir):
    ds = RepeatedData(get_data(datadir), -1)
    ds.reset_state()
    for ims, edgemaps in ds.get_data():
        for im, edgemap in zip(ims, edgemaps):
            print(im.shape)
            print(edgemap.shape)
            cv2.imshow("im", im )
            cv2.imshow("edge", edgemap)
            cv2.waitKey(0)

def get_config():
    logger.auto_set_dir()
    dataset_train = get_data(DATA_PATH)
    steps_per_epoch = dataset_train.size() * 10
    return TrainConfig(
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(keep_checkpoint_every_n_hours=0.1),
            ScheduledHyperParamSetter('learning_rate', [(10, 5e-4),(20,5e-5),(30,1e-5)]),
            HumanHyperParamSetter('learning_rate'),
        ],
        model=Unet(),
        steps_per_epoch=steps_per_epoch,
        max_epoch=100,
        )


def train():
    config = get_config()
    #config.session_init = get_model_loader('./train_log/train/model-3600.data-00000-of-00001')
    launch_train_with_config(
            config,
            SyncMultiGPUTrainer([0]))

def test(datadir,model_path):
    pred_config = PredictConfig(
        model=Unet(training=False),
        session_init=get_model_loader(model_path),
        input_names=['input'],
        output_names=['output'])
    predictor = OfflinePredictor(pred_config)
    data_dict = get_data_dict(datadir)
    print('Samples: ', [{label: len(data_dict[label]['filelist'])} for label in data_dict.keys()])
    rightN=0
    for label in data_dict.keys():
        data = data_dict[label]
        v_vec=[]
        for fname in data['filelist']:
            image = cv2.imread(fname)
            # im = cv2.resize(image,(960,1280))
            # im = np.expand_dims(im, 0).astype('float32')
            # outs = predictor(im)[0]
            # v = np.mean(outs[0])
            v=0.0
            row=0
            while(row<image.shape[0]):
                col=0
                while(col<image.shape[1]):
                    im = image[row:row+512, col:col+512]
                    im = cv2.resize(im,(512,512))
                    im = np.expand_dims(im,0).astype('float32')
                    c= np.mean(predictor(im)[0])
                    v+=c
                    col+=512
                row+=470
            # print(fname+' v: '+str(v))
            pred_label='正常'
            if v > 0:
                pred_label='不正常'
            if label=='正常':
                if pred_label==label:
                    rightN+=1
            else:
                if pred_label!='正常':
                    rightN+=1
            v_vec.append(v)
        v_vec = np.array(v_vec)
        print('label: '+ label + ' vmean: '+ str(np.mean(v_vec))
              + ' vmin: '+str(np.min(v_vec)) + ' vmax: '+str(np.max(v_vec)) )
    print('right n: ',rightN)

if __name__ =='__main__':
    #view_data('/home/lhw/Dataset/ali_flaw/haha')
    #test('/home/lhw/Dataset/ali_flaw/20180628',model_path='train_log/train/model-19090')
    train()
