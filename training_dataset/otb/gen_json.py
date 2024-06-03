#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import json
from os.path import join, exists
import os
import pandas as pd

# dataset_path = 'D:\\result\\siamGAT\\training_dataset\\data\\OTB100'
dataset_path= './data/otb'
train_sets=['Basketball','Car2','Dog','Human3','MotorRolling','Surfer','Biker','Car24','Dog1','Human4-2','MountainBike',
          'Suv','Bird1','Car4','Doll','Human5','Sylvester','Bird2','CarDark','DragonBaby','Human6','Tiger1','BlurBody',
          'CarScale','Dudek','Human7','Panda','Tiger2','BlurCar1','ClifBar','FaceOcc1','Human8','RedTeam','Toy','BlurCar2',
          'Coke','FaceOcc2','Human9','Rubik','Trans','BlurCar3','Couple','Fish','Ironman','Shaking','Trellis','BlurCar4',
          'Coupon','FleetFace','Jogging-1','Singer1','Twinnings','BlurFace','Crossing','Football','Jogging-2','Singer2','Vase',
          'BlurOwl','Crowds','Football1','Jump','Skater','Walking','Board','Dancer','Freeman1','Jumping','Skater2','Walking2','Bolt',
          'Dancer2','Freeman3','KiteSurf','Skating1','Woman','Bolt2','David','Freeman4','Lemming','Skating2-1','Box','David2',
          'Girl','Liquor','Skating2-2','Boy','David3','Girl2','Man','Skiing','Deer','Gym','Matrix','Soccer','Car1',
          'Diving','Human2','Mhyang','Subway']
# train_sets = ['GOT-10k_Train_split_01','GOT-10k_Train_split_02','GOT-10k_Train_split_03','GOT-10k_Train_split_04',
#             'GOT-10k_Train_split_05','GOT-10k_Train_split_06','GOT-10k_Train_split_07','GOT-10k_Train_split_08',
#             'GOT-10k_Train_split_09','GOT-10k_Train_split_10','GOT-10k_Train_split_11','GOT-10k_Train_split_12',
#             'GOT-10k_Train_split_13','GOT-10k_Train_split_14','GOT-10k_Train_split_15','GOT-10k_Train_split_16',
#             'GOT-10k_Train_split_17','GOT-10k_Train_split_18','GOT-10k_Train_split_19']
# val_set = ['val']
# d_sets = {'videos_val':val_set,'videos_train':train_sets}
d_sets = {'videos_train':train_sets}
# videos_val = ['MOT17-02-DPM']
# videos_train = ['MOT17-04-DPM','MOT17-05-DPM','MOT17-09-DPM','MOT17-11-DPM','MOT17-13-DPM']
# d_sets = {'videos_val':videos_val,'videos_train':videos_train}

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
                
                # # //video = dataset+'/'+video
                # print(dataset_path,'and',video)
                gt_path = join(dataset_path, video, 'groundtruth_rect.txt')
                #print(video)
                # print(gt_path)
                f = open(gt_path, 'r')
                groundtruth = f.readlines()
                f.close()
                for idx, gt_line in enumerate(groundtruth):
                    #print(gt_path)
                    if '\t' in gt_line:
                        # print('yes')
                        gt_image = gt_line.strip().split('\t')
                    else:
                        gt_image = gt_line.strip().split(',')
                    frame = '%06d' % (int(idx))
                    obj = '%02d' % (int(0))

                    bbox = [int(float(gt_image[0])), int(float(gt_image[1])),
                            int(float(gt_image[0])) + int(float(gt_image[2])),
                            int(float(gt_image[1])) + int(float(gt_image[3]))]  # xmin,ymin,xmax,ymax
                    video_temp=video+'/img'
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
