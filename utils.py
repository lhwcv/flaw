import tensorflow as tf
import  tensorflow.contrib.slim as slim
import xml.etree.ElementTree as ET
import os

def extra_conv_arg_scope(weight_decay=1e-5, activation_fn=None, normalizer_fn=None):
    with slim.arg_scope(
            [slim.conv2d, slim.conv2d_transpose],
            padding='SAME',
            weights_regularizer=slim.l2_regularizer(weight_decay),
            activation_fn=activation_fn,
            normalizer_fn=normalizer_fn, ) as arg_sc:
        with slim.arg_scope(
                [slim.fully_connected],
                weights_regularizer=slim.l2_regularizer(weight_decay),
                activation_fn=activation_fn,
                normalizer_fn=normalizer_fn) as arg_sc:
            return arg_sc




def get_data_dict(path,exclude_label=None):
    def read_filename_dict(path):
        filename_dict = {}
        for dir in os.listdir(path):
            jpg_files = []
            for file in os.listdir(os.path.join(path, dir)):
                if 'jpg' in file and 'gui' not in file:
                    jpg_files.append(os.path.join(path, dir, file))
            filename_dict[dir] = {'jpg_files': jpg_files}
        return filename_dict

    def parse_xml(xml_file):
        if not os.path.isfile(xml_file):
            return {'size': [], 'bboxes': [{'label': '正常', 'bb': []}]}
        tree = ET.parse(xml_file)
        root = tree.getroot()
        size = root.find('size')
        image_size = [int(size.find('height').text), int(size.find('width').text)]
        objects = root.findall('object')
        bboxes = []
        for obj in objects:
            bb = obj.find('bndbox')
            bbox = [int(bb.find('xmin').text),
                    int(bb.find('ymin').text),
                    int(bb.find('xmax').text),
                    int(bb.find('ymax').text)]
            bbox[0] = 0 if bbox[0] < 0 else bbox[0]
            bbox[1] = 0 if bbox[1] < 0 else bbox[1]
            bbox[2] = 0 if bbox[2] < 0 else bbox[2]
            bbox[3] = 0 if bbox[3] < 0 else bbox[3]
            bbox[0] = image_size[1]-1 if bbox[0] >= image_size[1] else bbox[0]
            bbox[1] = image_size[0]-1 if bbox[1] >= image_size[0] else bbox[1]
            bbox[2] = image_size[1]-1 if bbox[2] >= image_size[1] else bbox[2]
            bbox[3] = image_size[0]-1 if bbox[3] >= image_size[0] else bbox[3]
            label = obj.find('name').text
            bboxes.append({'label': label, 'bb': bbox})
        return {'size': image_size, 'bboxes': bboxes}
    filename_dict = read_filename_dict(path)
    data_dict = {}
    for label in filename_dict.keys():
        if exclude_label is not None:
            if label in exclude_label:
                continue
        label_samples_data = {}
        for jpg_file in filename_dict[label]['jpg_files']:
            xml_file = jpg_file.replace('jpg', 'xml')
            if not label == '正常' and not os.path.isfile(xml_file):
                print(xml_file + '　Not found!!!')
                exit(0)
            one_sample_data = parse_xml(xml_file)
            label_samples_data[jpg_file] = one_sample_data
        data_dict[label] = {'filelist': filename_dict[label]['jpg_files'], 'anno': label_samples_data}
    return data_dict

def calc_iou(x1, y1, width1, height1, x2, y2, width2, height2):
    endx = max(x1 + width1, x2 + width2)
    startx = min(x1, x2)
    width = width1 + width2 - (endx - startx)
    endy = max(y1 + height1, y2 + height2)
    starty = min(y1, y2)
    height = height1 + height2 - (endy - starty)
    Area = width * height  # 两矩形相交面积
    Area1 = width1 * height1
    Area2 = width2 * height2
    if width <= 0 or height <= 0:
        ratio = 0  # 重叠率为 0
        Area=0
        return ratio,Area,Area1,Area2
    else:

        ratio = Area * 1. / (Area1 + Area2 - Area)
        return ratio,Area,Area1,Area2