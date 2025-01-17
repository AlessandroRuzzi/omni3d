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
import json
from dataset import behave_camera_utils as bcu

from tqdm import tqdm

def do_test(args, cfg, model):

    import json
    
    with open('/data/aruzzi/Behave/info.json', 'r') as f:
        info = json.load(f)

    list_of_ims = info['img_paths'] # util.list_files(os.path.join(args.input_folder, ''), '*.jpg')
    list_of_bbx = info['bbox']
    list_of_verts = info['human_verts']
    
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
    human_predicted = {}

    intrinsics = [bcu.load_intrinsics(os.path.join("/data/xiwang/behave/calibs", "intrinsics"), i) for i in range(4)]
    
    for bbox, path, verts in zip(list_of_bbx, tqdm(list_of_ims), list_of_verts):

        im_name = path[len('/data/xiwang/behave/sequences/'):]

        im = util.imread(path)
        
        bbox[0] *= -1
        bbox[1] *= -1

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
        #K = np.array([
        #    [976.2120971679688, 0.0, 1017.9580078125], 
        #    [0.0, 976.0467529296875, 787.3128662109375], 
        #    [0.0, 0.0, 1.0]
        #])
        kid = int((im_name.split("/")[-1]).split(".")[0][1])
        
        K = intrinsics[kid][0]

        aug_input = T.AugInput(im)
        _ = augmentations(aug_input)
        image = aug_input.image

        batched = [{
            'image': torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))).cuda(), 
            'height': image_shape[0], 'width': image_shape[1], 'K': K
        }]

        dets = model(batched)[0]['instances']

        verts = torch.FloatTensor(verts)
        human_center = [(torch.min(verts[:,0]) + (torch.max(verts[:,0]) - torch.min(verts[:,0])) / 2.0).detach().cpu().float(), 
        (torch.min(verts[:,1]) + (torch.max(verts[:,1]) - torch.min(verts[:,1])) / 2.0).detach().cpu().float(),
        (torch.min(verts[:,2]) + (torch.max(verts[:,2]) - torch.min(verts[:,2])) / 2.0).detach().cpu().float()]
        
        human_center = [float(i) for i in human_center]

        bbox_project = human_center
        
        # find the sample with maximum dets.score
        res[im_name] = {
            "gt_bbox_center": bbox[:3],
            "gt_bbox_size": bbox[3:] * 3,
            "human_center": bbox_project,
            "human_dim": [float((torch.max(verts[:,0]) - torch.min(verts[:,0])).detach().cpu().numpy()), float((torch.max(verts[:,1]) - torch.min(verts[:,1])).detach().cpu().numpy()), float((torch.max(verts[:,2]) - torch.min(verts[:,2])).detach().cpu().numpy())]
        }

        dets_no_person = {}
        dets_person = {}

        dets_no_person["pred_center_cam"] = []
        dets_person["pred_center_cam"] = []
        dets_no_person["pred_dimensions"] = []
        dets_person["pred_dimensions"] = []
        dets_no_person["scores"] = []
        dets_person["scores"] = []
        dets_no_person["pred_classes"] = []
        dets_person["pred_classes"] = []
        dets_no_person["pred_pose"] = []
        dets_person["pred_pose"] = []
       
        for i in range(len(dets.scores)):
            if cats[dets.pred_classes[i].item()] == "person":
                    dets_person["pred_center_cam"].append(dets.pred_center_cam[i].tolist())
                    dets_person["pred_dimensions"].append(dets.pred_dimensions[i].tolist())
                    dets_person["scores"].append(dets.scores[i].tolist())
                    dets_person["pred_classes"].append(dets.pred_classes[i].tolist())
                    dets_person["pred_pose"].append(dets.pred_pose[i].tolist())
            elif cats[dets.pred_classes[i].item()] == "interaction":
                 continue
            else:
                    dets_no_person["pred_center_cam"].append(dets.pred_center_cam[i].tolist())
                    dets_no_person["pred_dimensions"].append(dets.pred_dimensions[i].tolist())
                    dets_no_person["scores"].append(dets.scores[i].tolist())
                    dets_no_person["pred_classes"].append(dets.pred_classes[i].tolist())
                    dets_no_person["pred_pose"].append(dets.pred_pose[i].tolist())               

        n_det_objects = len(dets_no_person["scores"])
        n_det_person = len(dets_person["scores"])

        if n_det_objects > 0:
            max_idx = torch.argmax(torch.FloatTensor(dets_no_person["scores"])).item()
            res[im_name]['pred_bbox_center'] = dets_no_person["pred_center_cam"][max_idx]
            res[im_name]['pred_bbox_size'] = dets_no_person["pred_dimensions"][max_idx]
            res[im_name]['pred_bbox_score'] = dets_no_person["scores"][max_idx]
            res[im_name]['pred_bbox_class'] = cats[dets_no_person["pred_classes"][max_idx]]
            res[im_name]['pred_bbox_orientation'] = dets_no_person["pred_pose"][max_idx]
            
            all_predicted[im_name] = {
                "bbox_center": dets_no_person["pred_center_cam"],
                "bbox_size": dets_no_person["pred_dimensions"],
                "bbox_score": dets_no_person["scores"],
                "bbox_class": [cats[c] for c in dets_no_person["pred_classes"]],
                "bbox_orientation": dets_no_person["pred_pose"]
            }
        if n_det_person > 0:
            human_predicted[im_name] = {}
            max_idx = torch.argmax(torch.FloatTensor(dets_person["scores"])).item()
            human_predicted[im_name]['pred_bbox_center'] = dets_person["pred_center_cam"][max_idx]
            human_predicted[im_name]['pred_bbox_size'] = dets_person["pred_dimensions"][max_idx]
            human_predicted[im_name]['pred_bbox_score'] = dets_person["scores"][max_idx]
            human_predicted[im_name]['pred_bbox_class'] = cats[dets_person["pred_classes"][max_idx]]
            human_predicted[im_name]['pred_bbox_orientation'] = dets_person["pred_pose"][max_idx]

        #break

    with open('predictions/results_interaction_test_2.json', 'w') as f:
        json.dump({"best_score vs gt": res, "all_predicted": all_predicted, "person": human_predicted}, f)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    get_cfg_defaults(cfg)

    config_file = args.config_file

    # store locally if needed
    if config_file.startswith(util.CubeRCNNHandler.PREFIX):    
        config_file = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, config_file)

    cfg.merge_from_file(config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)
    model = build_model(cfg)
    
    logger.info("Model:\n{}".format(model))
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=True
    )

    with torch.no_grad():
        do_test(args, cfg, model)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        epilog=None, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="/configs/base_Behave.yaml", metavar="FILE", help="path to config file")
    parser.add_argument('--input-folder',  type=str, help='list of image folders to process', required=True)
    parser.add_argument("--threshold", type=float, default=0.25, help="threshold on score for visualizing")
    parser.add_argument("--display", default=False, action="store_true", help="Whether to show the images in matplotlib",)
    
    parser.add_argument("--eval-only", default=True, action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. "
        "See config references at "
        "https://detectron2.readthedocs.io/modules/config.html#config-references",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

