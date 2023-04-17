import wandb
from glob import glob
import cv2
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.data import transforms as T
import os
import argparse
import sys
from cubercnn.config import get_cfg_defaults
from cubercnn.modeling.proposal_generator import RPNWithIgnore
from cubercnn.modeling.roi_heads import ROIHeads3D
from cubercnn.modeling.meta_arch import RCNN3D, build_model
from cubercnn.modeling.backbone import build_dla_from_vision_fpn_backbone
from cubercnn import util, vis
from cubercnn.vis import draw_3d_box, draw_text, draw_scene_view
import math
import logging
logger = logging.getLogger("detectron2")
from dataset import behave_camera_utils as bcu
from PIL import Image
from operator import itemgetter
import numpy as np

wandb.init(project = "Omni3D")

category = [ {'id' : 0, 'name' : 'backpack', 'supercategory' : ""}, {'id' : 1, 'name' :'basketball', 'supercategory' : ""}, {'id' : 2, 'name' :'boxlarge', 'supercategory' : ""}, 
             {'id' : 3, 'name' :'boxlong', 'supercategory' : ""}, {'id' : 4, 'name' :'boxmedium', 'supercategory' : ""}, {'id' : 5, 'name' :'boxsmall', 'supercategory' : ""}, 
             {'id' : 6, 'name' :'boxtiny', 'supercategory' : ""}, {'id' : 7, 'name' :'chairblack', 'supercategory' : ""},{'id' : 8, 'name' :'chairwood', 'supercategory' : ""},
             {'id' : 9, 'name' :'keyboard', 'supercategory' : ""},{'id' : 10, 'name' :'monitor', 'supercategory' : ""}, {'id' : 11, 'name' :'plasticcontainer', 'supercategory' : ""}, 
             {'id' : 12, 'name' :'stool', 'supercategory' : ""}, {'id' : 13, 'name' :'suitcase', 'supercategory' : ""}, {'id' : 14, 'name' :'tablesmall', 'supercategory' : ""}, 
             {'id' : 15, 'name' :'tablesquare', 'supercategory' : ""}, {'id' : 16, 'name' :'toolbox', 'supercategory' : ""}, {'id' : 17, 'name' :'trashbin', 'supercategory' : ""}, 
             {'id' : 18, 'name' :'yogaball', 'supercategory' : ""}, {'id' : 19, 'name' :'yogamat', 'supercategory' : ""}, {'id' : 20, 'name' :'person', 'supercategory' : ""},
             {'id' : 21, 'name' :'interaction', 'supercategory' : ""}]

def log_bboxes(img,day, object_box, object_dim, object_orientation, object_cat, object_score, human_box, human_dim, human_orientation, human_score):
        intrinsics = [bcu.load_intrinsics(os.path.join("/data/xiwang/behave/calibs", "intrinsics"), i) for i in range(4)]
        id_cat = None
        for j,elem in enumerate(category):
            if elem['name'] == object_cat:
                id_cat = j
                break
        
        kid = day
        K = intrinsics[kid][0]
        color = util.get_color(id_cat)

        meshes = []
        meshes_text = []
        bbox3D = object_box + object_dim

        meshes_text.append('{} {:.2f}'.format(object_cat, object_score))
        color = [c/255.0 for c in util.get_color(id_cat)]
        box_mesh = util.mesh_cuboid(bbox3D, object_orientation, color=color)
        meshes.append(box_mesh)

        bbox3D = human_box + human_dim

        meshes_text.append('{} {:.2f}'.format("human", human_score))
        color = [c/255.0 for c in util.get_color(20)]
        box_mesh = util.mesh_cuboid(bbox3D, human_orientation, color=color)
        meshes.append(box_mesh)

        im_drawn_rgb, im_topdown, _ = draw_scene_view(img, K, meshes, text=meshes_text, scale=img.shape[0], blend_weight=0.5, blend_weight_overlay=0.85)

        images = wandb.Image(im_drawn_rgb, caption="Frontal image with predicted 3D bounding boxes")
        wandb.log({"Pred BBox" : images})

        images = wandb.Image(im_topdown, caption="Topdown image with predicted 3D bounding boxes")
        wandb.log({"Pred BBox" : images})

  
        tmp_img = (np.concatenate([im_drawn_rgb, im_topdown], axis=1)).astype(np.uint8)
        final_log_image = Image.fromarray(tmp_img)
        images = wandb.Image(final_log_image, caption="Topdown image with predicted 3D bounding boxes")
        wandb.log({"Full Image" : images})

        return final_log_image

def test_intercap(args, cfg, model):
    path = '/data/xiwang/InterCap/RGBD_Images/01/01/Seg_0/Frames_Cam1/color'

    images_path_list = [x for x in glob("%s/*.jpg" % path)]

    model.eval()

    category_path = os.path.join(util.file_parts(args.config_file)[0], 'category_meta.json')
        
    # store locally if needed
    if category_path.startswith(util.CubeRCNNHandler.PREFIX):
        category_path = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, category_path)

    metadata = util.load_json(category_path)
    cats = metadata['thing_classes']

    for image_path in images_path_list:
        res = {}
        all_predicted = {}
        human_predicted = {}
        im_name = image_path
        img = cv2.imread(image_path)
        #torch_image = torch.FloatTensor(img)
        #torch_image = torch.reshape(torch_image , (1,torch_image.shape[0], torch_image.shape[1], torch_image.shape[2]))
        K = np.array([
            [976.2120971679688, 0.0, 1017.9580078125], 
            [0.0, 976.0467529296875, 787.3128662109375], 
            [0.0, 0.0, 1.0]
        ])

        batched = [{
            'image': torch.as_tensor(np.ascontiguousarray(img.transpose(2, 0, 1))).cuda(), 
            'height': img.shape[0], 'width': img.shape[1], 'K': K
        }]

        dets = model(batched)[0]['instances']

        res[im_name] = {}
        all_predicted[im_name] = {}
        human_predicted[im_name] = {}
        
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


        pred_dict = res[im_name]
        pred_all = all_predicted[im_name]
        
        #gt_box = pred_dict["gt_bbox_center"]
        #gt_length = pred_dict["gt_bbox_size"][0]
       
        try:
            pred_human= human_predicted[im_name]
            human_center = pred_human["pred_bbox_center"]

            object_dist_list = []
            for i, bbox in enumerate(pred_all["bbox_center"]):
                #print("human distance: ",math.dist(human_center, bbox), " Confidence: ", (1-pred_all["bbox_score"][i]))
                object_dist_list.append(math.dist(human_center, bbox) + (1-pred_all["bbox_score"][i]))

            pos, element = min(enumerate(object_dist_list), key=itemgetter(1))
            pred_box = pred_all["bbox_center"][pos]
            pred_length = pred_all["bbox_size"][pos]
            pred_pose = pred_all["bbox_orientation"][pos]
            pred_cat = pred_all["bbox_class"][pos]
            pred_score = pred_all["bbox_score"][pos]
        except:
            #counter+=1
            pred_box = pred_dict["pred_bbox_center"]
            pred_length = pred_dict["pred_bbox_size"]
            pred_pose = pred_dict["pred_bbox_orientation"]
            pred_cat = pred_dict["pred_bbox_class"]
            pred_score = pred_dict["pred_bbox_score"]


        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_wandb = log_bboxes(img, 0, pred_box, pred_length, pred_pose, pred_cat, pred_score, human_center, pred_human["pred_bbox_size"], pred_human["pred_bbox_orientation"], pred_human["pred_bbox_score"])



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
        test_intercap(args, cfg, model)

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

