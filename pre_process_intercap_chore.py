import sys, os
import numpy as np
sys.path.append(os.getcwd())
import trimesh
from os.path import join, isfile

from mmdet.apis import init_detector, inference_detector
import mmcv
from mmdet.core.evaluation.class_names import coco_classes
import cv2
import json
import shutil
import pickle
import joblib
import csv
from glob import glob
import wandb 
sys.path.insert(0, '/local/home/aruzzi/')
sys.path.insert(0, '/local/home/aruzzi/openpose')
sys.path.insert(0, '/local/home/aruzzi/PyMAF/')
from PARE.pare.utils.geometry import batch_rot2aa
from psbody.mesh import Mesh
from PyMAF.models.smpl import get_smpl_faces
import torch

wandb.init("Intercap CHORE")

relation_dict = {"01": 28, "02": 36, "03": 32, "04": 25, "05": 38, "06": 26, "07": 56, "08": 39, "09": 41, "10": 56}


def log_mask(img_to_log, mask, description, class_labels):
    image_gt = wandb.Image(img_to_log, caption="Image")
    mask_img = wandb.Image(
                    image_gt,
                    masks={
                        "predictions": {
                            "mask_data": mask,
                            "class_labels": class_labels,
                        }
                    },
                )
    wandb.log({description: mask_img})


if __name__ == "__main__":

    BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

    POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]
    
    threshold = 0.2
    
    mmdet_root = "../mmdetection/"
    model_eval = "MRCNN"
    filter_method = "enlarge"
    model_eval = "_".join((model_eval, filter_method))

    if model_eval.startswith("MRCNN"):
        config_file = mmdet_root + "configs/mask_rcnn/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco.py"
        checkpoint_file = "checkpoint/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco_20210526_120447-c376f129.pth"

    elif model_eval.startswith("PointRend"):
        config_file = mmdet_root + "configs/point_rend/point_rend_r50_caffe_fpn_mstrain_3x_coco.py"
        checkpoint_file = mmdet_root + "checkpoints/point_rend_r50_caffe_fpn_mstrain_3x_coco-e0ebb6b7.pth"

    elif model_eval == "CCDRCNN":
        config_file = mmdet_root + "configs/cascade_rcnn/cascade_mask_rcnn_x101_64x4d_fpn_mstrain_3x_coco.py"
        checkpoint_file = mmdet_root + "checkpoints/cascade_mask_rcnn_x101_64x4d_fpn_mstrain_3x_coco_20210719_210311-d3e64ba0.pth"

    elif model_eval.startswith("Mask2Former"):
        config_file = mmdet_root + "configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
        checkpoint_file = mmdet_root + "checkpoints/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth"

    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    data_path = "/data/xiwang/InterCap/RGBD_Images"
    save_path = "/data/aruzzi/Intercap/"
    pkl_path = "/mnt/scratch/kexshi/PARE_Res/"
    humans = [f for f in os.listdir(data_path) if not(isfile(os.path.join(data_path, f)))]
    humans.sort()
    for human in humans:
        human_path = os.path.join(data_path, human)
        objects = [f for f in os.listdir(human_path) if not(isfile(os.path.join(human_path, f)))]
        objects.sort()
        for object in objects:
            object_path = os.path.join(human_path, object)
            images_path = os.path.join(object_path, "Seg_0/Frames_Cam1/color/")
            images_name = [x for x in glob("%s/*.jpg" % images_path)]
            images_name.sort()
            for image in images_name:

                #create folder

                final_folder_path = os.path.join(save_path, human+object+image.split("/")[-1][:-4] + "/")
                if os.path.exists(final_folder_path):
                    shutil.rmtree(final_folder_path, ignore_errors=True)
                os.makedirs(final_folder_path)

                #calculate masks

                try:
                    img = cv2.imread(image) 
                    res = inference_detector(model, img)

                    body_mask = res[1][0][0]
                    obj_mask = res[1][relation_dict[object]][0]
                except:
                    shutil.rmtree(final_folder_path, ignore_errors=True)
                    continue

                #convert pkl into ply and json

                pkl_path_image = os.path.join(pkl_path, human,object, "Seg_0/Frames_Cam1/" + image.split("/")[-1][-9:-4] + ".pkl")
                
                pare_pred = joblib.load(pkl_path_image)

               
                if pare_pred is None or pare_pred["smpl_vertices"] is None:                
                    print("Missing detection")                
                    continue          
                smpl_pred = Mesh(v=pare_pred["smpl_vertices"][0], f=get_smpl_faces())            
                pred_pose = batch_rot2aa(torch.from_numpy(pare_pred["pred_pose"][0])).reshape(-1).numpy().tolist()           
                pred_shape = pare_pred["pred_shape"].reshape(-1).tolist()
                new_json = {"pose": pred_pose,"betas": pred_shape}     

                
                #openpose estimation
                shutil.copyfile(image, final_folder_path + "/k1.color.jpg")
                os.system(
                f'./build/examples/openpose/openpose.bin --image_dir {final_folder_path} --face --hand --write_json {final_folder_path}'
                 )
                
                #save files
                cv2.imwrite(final_folder_path + "/k1.person_mask.jpg", body_mask * 225)
                cv2.imwrite(final_folder_path + "/k1.obj_mask.jpg", obj_mask * 225)
                smpl_pred.write_ply(final_folder_path + "k1.mocap.ply")
                with open(final_folder_path + "k1.mocap.json", "w") as f:  
                     json.dump(new_json, f, indent=4)

                #log_mask(img, body_mask, "body mask", {0: "background", 255: "body"})
                #log_mask(img, obj_mask, "object mask", {0: "background", 255: "object"})

                break
            break
        break

    print("Intercap dataset processed!")
