from os.path import join, isdir
from os import listdir, mkdir, makedirs
import cv2
import numpy as np
import glob
import xml.etree.ElementTree as ET
from concurrent import futures
import sys
import time
import os
# dataset_path = 'D:\\result\\siamGAT\\training_dataset\\data\\OTB100'
dataset_path= './data/otb'
# sub_sets=['Basketball','Car2','Dog','Human3','MotorRolling','Surfer','Biker','Car24','Dog1','Human4-2','MountainBike',
#           'Suv','Bird1','Car4','Doll','Human5','Sylvester','Bird2','CarDark','DragonBaby','Human6','Tiger1','BlurBody',
#           'CarScale','Dudek','Human7','Panda','Tiger2','BlurCar1','ClifBar','FaceOcc1','Human8','RedTeam','Toy','BlurCar2',
#           'Coke','FaceOcc2','Human9','Rubik','Trans','BlurCar3','Couple','Fish','Ironman','Shaking','Trellis','BlurCar4',
#           'Coupon','FleetFace','Jogging-1','Singer1','Twinnings','BlurFace','Crossing','Football','Jogging-2','Singer2','Vase',
#           'BlurOwl','Crowds','Football1','Jump','Skater','Walking','Board','Dancer','Freeman1','Jumping','Skater2','Walking2','Bolt',
#           'Dancer2','Freeman3','KiteSurf','Skating1','Woman','Bolt2','David','Freeman4','Lemming','Skating2-1','Box','David2',
#           'Girl','Liquor','Skating2-2','Boy','David3','Girl2','Man','Skiing','Deer','Gym','Matrix','Soccer','Car1',
#           'Diving','Human2','Mhyang','Subway']
sub_sets=['BlurCar4']#,'BlurCar3','BlurCar4']
# ['Basketball','Car2','Dog','Human3','MotorRolling','Surfer',
# 'Biker','Car24','Dog1','Human4-2','MountainBike','Suv',
# 'Bird1','Car4','Doll',',Human5','Sylvester',
# 'Bird2','CarDark','DragonBaby','Human6','Tiger1',
# BlurBody/    CarScale/  Dudek/       Human7/     Panda/         Tiger2/
# BlurCar1/    ClifBar/   FaceOcc1/    Human8/     RedTeam/       Toy/
# BlurCar2/    Coke/      FaceOcc2/    Human9/     Rubik/         Trans/
# BlurCar3/    Couple/    Fish/        Ironman/    Shaking/       Trellis/
# BlurCar4/    Coupon/    FleetFace/   Jogging-1/  Singer1/       Twinnings/
# BlurFace/    Crossing/  Football/    Jogging-2/  Singer2/       Vase/
# BlurOwl/     Crowds/    Football1/   Jump/       Skater/        Walking/
# Board/       Dancer/    Freeman1/    Jumping/    Skater2/       Walking2/
# Bolt/        Dancer2/   Freeman3/    KiteSurf/   Skating1/      Woman/
# Bolt2/       David/     Freeman4/    Lemming/    Skating2-1/
# Box/         David2/    Girl/        Liquor/     Skating2-2/
# Boy/         David3/    Girl2/       Man/        Skiing/
# CVPR13.json  Deer/      Gym/         Matrix/     Soccer/
# Car1/        Diving/    Human2/      Mhyang/     ,'Subway']
# Print iterations progress (thanks StackOverflow)
def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr       = "{0:." + str(decimals) + "f}"
    percents        = formatStr.format(100 * (iteration / float(total)))
    filledLength    = int(round(barLength * iteration / float(total)))
    bar             = '' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\x1b[2K\r')
    sys.stdout.flush()


def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz-1) / (bbox[2]-bbox[0])
    b = (out_sz-1) / (bbox[3]-bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float32)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop


def pos_s_2_bbox(pos, s):
    return [pos[0]-s/2, pos[1]-s/2, pos[0]+s/2, pos[1]+s/2]


def crop_like_SiamFC(image, bbox, context_amount=0.5, exemplar_size=127, instanc_size=255, padding=(0, 0, 0)):
    target_pos = [(bbox[2]+bbox[0])/2., (bbox[3]+bbox[1])/2.]
    target_size = [bbox[2]-bbox[0], bbox[3]-bbox[1]]
    wc_z = target_size[1] + context_amount * sum(target_size)
    hc_z = target_size[0] + context_amount * sum(target_size)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = exemplar_size / s_z
    d_search = (instanc_size - exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    z = crop_hwc(image, pos_s_2_bbox(target_pos, s_z), exemplar_size, padding)
    x = crop_hwc(image, pos_s_2_bbox(target_pos, s_x), instanc_size, padding)
    return z, x


def crop_video(video, d_set, crop_path, instanc_size):
    if video != 'list.txt':
        # video_crop_base_path = join(crop_path, video)
        video_crop_base_path = join(crop_path, 'img')
        if not isdir(video_crop_base_path): makedirs(video_crop_base_path)
        gt_path = join(dataset_path, d_set, 'groundtruth_rect.txt')
       
        images_path = join(dataset_path, d_set, 'img')
        
        f = open(gt_path, 'r')
        groundtruth = f.readlines()
        f.close()
        for idx, gt_line in enumerate(groundtruth):
            
            if '\t' in gt_line:
                # print('yes')
                gt_image = gt_line.strip().split('\t')
            else:
                gt_image = gt_line.strip().split(',')
            bbox = [int(float(gt_image[0])),int(float(gt_image[1])),int(float(gt_image[0]))+int(float(gt_image[2])),int(float(gt_image[1]))+int(float(gt_image[3]))]#xmin,ymin,xmax,ymax

            im = cv2.imread(join(images_path,str(idx+1).zfill(4)+'.jpg'))
            avg_chans = np.mean(im, axis=(0, 1))

            z, x = crop_like_SiamFC(im, bbox, instanc_size=instanc_size, padding=avg_chans)
            cv2.imwrite(join(video_crop_base_path, '{:06d}.{:02d}.z.jpg'.format(int(idx), int(0))), z)
            cv2.imwrite(join(video_crop_base_path, '{:06d}.{:02d}.x.jpg'.format(int(idx), int(0))), x)


def main(instanc_size=511, num_threads=1):
    # path='./'
    crop_path = './crop{:d}'.format(instanc_size)
    # crop_path=os.path.join(path,crop_path)
    if not isdir(crop_path): mkdir(crop_path)
    for d_set in sub_sets:
        save_path = join(crop_path, d_set)
        videos = listdir(join(dataset_path,d_set))
        if not isdir(save_path): mkdir(save_path)


        n_videos = len(videos)
        with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
            fs = [executor.submit(crop_video, video, d_set, save_path, instanc_size) for video in videos]
            for i, f in enumerate(futures.as_completed(fs)):
                # Write progress to error so that it can be seen
                printProgress(i, n_videos, prefix='train', suffix='Done ', barLength=40)


if __name__ == '__main__':
    since = time.time()
    main()
    time_elapsed = time.time() - since
    print('Total complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
