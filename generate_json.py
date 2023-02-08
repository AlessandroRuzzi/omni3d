from dataset.behave_dataset import BehaveImgDataset

# make an options object
class Options():
    def __init__(self):
        self.max_dataset_size = 5000000
        self.trunc_thres = 0.2

opt = Options()

dataset = BehaveImgDataset()
dataset.initialize(opt, 'train')

import os

bbox = []
img_paths = []

for index, i in enumerate(dataset):
    # copy the image (i['img_path']) to a new folder (/home/cluster/workshop/omni3d/datasets/miniBEHAVE)
    # name = i['img_path'][len('/home/cluster/workshop/data/behave/sequences/'):]
    # name = name.replace('/', '_')
    if index % 1000 == 0:
        print(index)
    
    bbox.append(i['bbox'].tolist())
    img_paths.append(i['img_path'])
    

# dump 'bbox' to a json file

import json

with open('/data/aruzzi/Behave/train_info.json', 'w') as f:
    json.dump({"bbox": bbox, "img_paths": img_paths}, f)