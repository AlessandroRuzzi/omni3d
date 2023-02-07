# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
import os
import argparse
import sys
import numpy as np
from collections import OrderedDict
import torch
from itertools import chain

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.data import transforms as T

logger = logging.getLogger("detectron2")

sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

from cubercnn.config import get_cfg_defaults
from cubercnn.modeling.proposal_generator import RPNWithIgnore
from cubercnn.modeling.roi_heads import ROIHeads3D
from cubercnn.modeling.meta_arch import RCNN3D, build_model
from cubercnn.modeling.backbone import build_dla_from_vision_fpn_backbone
from cubercnn import util, vis

from tqdm import tqdm

def do_test(args, cfg, model):

    import json
    
    with open('/data/aruzzi/Behave/Behave_test.json', 'r') as f:
        info = json.load(f)

    list_of_ims = info['img_paths'] # util.list_files(os.path.join(args.input_folder, ''), '*.jpg')
    list_of_bbx = info['bbox']
    
    model.eval()

    thres = args.threshold

    output_dir = cfg.OUTPUT_DIR
    min_size = cfg.INPUT.MIN_SIZE_TEST
    max_size = cfg.INPUT.MAX_SIZE_TEST
    augmentations = T.AugmentationList([T.ResizeShortestEdge(min_size, max_size, "choice")])

    util.mkdir_if_missing(output_dir)

    category_path = os.path.join(util.file_parts(args.config_file)[0], 'category_meta.json')
        
    # store locally if needed
    if category_path.startswith(util.CubeRCNNHandler.PREFIX):
        category_path = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, category_path)

    metadata = util.load_json(category_path)
    cats = metadata['thing_classes']
    
    res = {}
    all_predicted = {}
    
    for bbox, path in zip(list_of_bbx, tqdm(list_of_ims)):

        im_name = path[len('/home/cluster/workshop/data/behave/sequences/'):]

        im = util.imread(path)
        
        #bbox[0] *= -1
        #bbox[1] *= -1

        if im is None:
            continue
        
        image_shape = im.shape[:2]  # h, w

        # h, w = image_shape
        # f_ndc = 4
        # f = f_ndc * h / 2

        # K = np.array([
        #     [f, 0.0, w/2], 
        #     [0.0, f, h/2], 
        #     [0.0, 0.0, 1.0]
        # ])
        K = np.array([
            [976.2120971679688, 0.0, 1017.9580078125], 
            [0.0, 976.0467529296875, 787.3128662109375], 
            [0.0, 0.0, 1.0]
        ])

        aug_input = T.AugInput(im)
        _ = augmentations(aug_input)
        image = aug_input.image

        batched = [{
            'image': torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))).cuda(), 
            'height': image_shape[0], 'width': image_shape[1], 'K': K
        }]

        dets = model(batched)[0]['instances']
        n_det = len(dets)

        meshes = []
        meshes_text = []
        
        # find the sample with maximum dets.score
        res[im_name] = {
            "gt_bbox_center": bbox[:3],
            "gt_bbox_size": bbox[3:] * 3,
        }

        if n_det > 0:
            max_idx = torch.argmax(dets.scores).item()
            res[im_name]['pred_bbox_center'] = dets.pred_center_cam[max_idx].tolist()
            res[im_name]['pred_bbox_size'] = dets.pred_dimensions[max_idx].tolist()
            res[im_name]['pred_bbox_score'] = dets.scores[max_idx].item()
            res[im_name]['pred_bbox_class'] = cats[dets.pred_classes[max_idx].item()]
            res[im_name]['pred_bbox_orientation'] = dets.pred_pose[max_idx].tolist()
            
            all_predicted[im_name] = {
                "bbox_center": dets.pred_center_cam.tolist(),
                "bbox_size": dets.pred_dimensions.tolist(),
                "bbox_score": dets.scores.tolist(),
                "bbox_class": [cats[c] for c in dets.pred_classes.tolist()],
                "bbox_orientation": dets.pred_pose.tolist()
            }

if __name__ == "__main__":
    do_test()

