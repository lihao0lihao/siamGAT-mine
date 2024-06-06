#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import json
from os.path import join, exists
import os
import pandas as pd

# dataset_path = 'D:\\result\\siamGAT\\training_dataset\\data\\OTB100'
dataset_path= './data/vot2018_lt'
train_sets=['ballet',   'bird1',  'car3',  'car9', 'cat2','freestyle',  'group3',     'nissan','person19',
            'person4',  'rollerman',  'uav1',
'bicycle',  'car1','car6','carchase','dragon','group1','liverRun', 'person14','person2','person5','skiing',     
'bike1','car16', 'car8','cat1','following',  'group2','longboard','person17',  'person20',  'person7',
'tightrope','yamaha']

d_sets = {'videos_train':train_sets}


def parse_and_sched(dl_dir='.'):
    # For each of the two datasets
    js = {}
    for d_set in d_sets:
        for dataset in d_sets[d_set]:
            # print('dataset,',dataset)
            videos = os.listdir(os.path.join(dataset_path))
            # print(videos)
            for video in videos:
                if video == 'list.txt':
                    continue
                if video =='VOT2018-LT.json':
                    continue

                # # //video = dataset+'/'+video
                # print(dataset_path,'and',video)
                gt_path = join(dataset_path, video, 'groundtruth.txt')
                #print(video)
                #print(gt_path)
                f = open(gt_path, 'r')
                groundtruth = f.readlines()
                f.close()
                idx_step=0
                for idx, gt_line in enumerate(groundtruth):
                    #print(gt_path,"   ",gt_line)
                    if 'nan' in gt_line:
                        continue
                    
                    if '\t' in gt_line:
                        # print('yes')
                        gt_image = gt_line.strip().split('\t')
                    else:
                        gt_image = gt_line.strip().split(',')
                    frame = '%06d' % (int(idx_step))
                    idx_step=idx_step+1
                    obj = '%02d' % (int(0))
                    
                    bbox = [int(float(gt_image[0])), int(float(gt_image[1])),
                            int(float(gt_image[0])) + int(float(gt_image[2])),
                            int(float(gt_image[1])) + int(float(gt_image[3]))]  # xmin,ymin,xmax,ymax
                    video_temp=video+'/color'
                    if video_temp not in js:
                        js[video_temp] = {}
                    if obj not in js[video_temp]:
                        js[video_temp][obj] = {}
                    js[video_temp][obj][frame] = bbox
        if 'videos_val' == d_set:
            json.dump(js, open('val.json', 'w'), indent=4, sort_keys=True)
        else:
            #path='D:\\result\\siamGAT\\training_dataset\\otb\\train.json'
            json.dump(js, open('train.json', 'w'), indent=4, sort_keys=True)
        js = {}

        print(d_set+': All videos downloaded' )


if __name__ == '__main__':
    parse_and_sched()
