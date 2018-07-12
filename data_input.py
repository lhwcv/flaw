import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
from src.utils import *
# def get_data_dict(path,exclude_label=None):
#     def read_filename_dict(path):
#         filename_dict = {}
#         for dir in os.listdir(path):
#             jpg_files = []
#             for file in os.listdir(os.path.join(path, dir)):
#                 if 'jpg' in file and 'gui' not in file:
#                     jpg_files.append(os.path.join(path, dir, file))
#             filename_dict[dir] = {'jpg_files': jpg_files}
#         return filename_dict
#
#     def parse_xml(xml_file):
#         if not os.path.isfile(xml_file):
#             return {'size': [], 'bboxes': [{'label': '正常', 'bb': []}]}
#         tree = ET.parse(xml_file)
#         root = tree.getroot()
#         size = root.find('size')
#         image_size = [int(size.find('height').text), int(size.find('width').text)]
#         objects = root.findall('object')
#         bboxes = []
#         for obj in objects:
#             bb = obj.find('bndbox')
#             bbox = [int(bb.find('xmin').text),
#                     int(bb.find('ymin').text),
#                     int(bb.find('xmax').text),
#                     int(bb.find('ymax').text)]
#             bbox[0] = 0 if bbox[0] < 0 else bbox[0]
#             bbox[1] = 0 if bbox[1] < 0 else bbox[1]
#             bbox[2] = 0 if bbox[2] < 0 else bbox[2]
#             bbox[3] = 0 if bbox[3] < 0 else bbox[3]
#             bbox[0] = image_size[1]-1 if bbox[0] >= image_size[1] else bbox[0]
#             bbox[1] = image_size[0]-1 if bbox[1] >= image_size[0] else bbox[1]
#             bbox[2] = image_size[1]-1 if bbox[2] >= image_size[1] else bbox[2]
#             bbox[3] = image_size[0]-1 if bbox[3] >= image_size[0] else bbox[3]
#             label = obj.find('name').text
#             bboxes.append({'label': label, 'bb': bbox})
#         return {'size': image_size, 'bboxes': bboxes}
#     filename_dict = read_filename_dict(path)
#     data_dict = {}
#     for label in filename_dict.keys():
#         if exclude_label is not None:
#             if label in exclude_label:
#                 continue
#         label_samples_data = {}
#         for jpg_file in filename_dict[label]['jpg_files']:
#             xml_file = jpg_file.replace('jpg', 'xml')
#             if not label == '正常' and not os.path.isfile(xml_file):
#                 print(xml_file + '　Not found!!!')
#                 exit(0)
#             one_sample_data = parse_xml(xml_file)
#             label_samples_data[jpg_file] = one_sample_data
#         data_dict[label] = {'filelist': filename_dict[label]['jpg_files'], 'anno': label_samples_data}
#     return data_dict



def _test_gui(path):
    data_dict = get_data_dict(path)
    print('Samples: ', [{label: len(data_dict[label]['filelist'])} for label in data_dict.keys()])

    for label in data_dict.keys():
        data = data_dict[label]
        for fname in data['filelist']:
            image = cv2.imread(fname)
            try:
                for bb in data['anno'][fname]['bboxes']:
                    bb = bb['bb']
                    cv2.rectangle(image, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 5, 16)
            except:
                ## 正常的没有bb
                pass
            dst_fname = fname.replace('.jpg', '_gui.jpg')
            cv2.imwrite(dst_fname, image)

            # print(data['anno'][fname]['size'])
            # print(data['anno'][fname]['bboxes'])

def _is_abnormal_block(xywh, bboxes):
    iou_accu=0
    r_accu=0

    for b in bboxes:
        b = b['bb']
        iou,s,s1,s2=calc_iou(xywh[0],xywh[1],xywh[2],xywh[3],
                             b[0],b[1],abs(b[2]-b[0]),abs(b[3]-b[1]))
        iou_accu+=iou
        r_accu+=s/s2
    # print('iou: ',iou_accu)
    # print('r_accu',r_accu)
    return iou_accu,r_accu



def _construct_clc_data(path, dst_dir='/home/lhw/Dataset/ali_flaw/clc_data'):
    data_dict = get_data_dict(path)
    print('Samples: ', [{label: len(data_dict[label]['filelist'])} for label in data_dict.keys()])
    id=0

    rr=0
    gg=0
    bb=0

    for label in data_dict.keys():
        if label=='正常':
            continue
        data = data_dict[label]
        for fname in data['filelist']:
            bboxes = data['anno'][fname]['bboxes']
            image = cv2.imread(fname)
            v=0.0
            row=0
            while(row<image.shape[0]):
                col=0
                while(col<image.shape[1]):
                    im = image[row:row+384, col:col+384]
                    im = cv2.resize(im,(384,384))
                    bb +=np.mean(im[:,:,0])
                    gg += np.mean(im[:, :, 1])
                    rr += np.mean(im[:, :, 2])


                    # if label!='正常':
                    #     iou, r = _is_abnormal_block([col, row, 384, 384], bboxes)
                    #     if r>0.2:
                    #         cv2.imwrite(dst_dir + '/abnormal/' + str(id) + '.jpg', im)
                    #     if r<0.001:
                    #         cv2.imwrite(dst_dir + '/normal/' + str(id) + '.jpg', im)
                    # else:
                    #     cv2.imwrite(dst_dir+'/normal/'+str(id)+'.jpg',im)
                    id+=1

                    col+=320
                row+=320
            #exit(0)
    print('bb: ',bb/id)
    print('gg: ', gg / id)
    print('rr: ', rr / id)





DATA_PATH='/home/lhw/Dataset/ali_flaw/20180705'

if __name__=='__main__':
    #_test_gui(DATA_PATH)
    _construct_clc_data(DATA_PATH)
    pass




