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
                    print(object)
                    print(relation_dict[object])
                    continue

                #convert pkl into ply and json

                
                pkl_path_image = os.path.join(pkl_path, human,object, "Seg_0/Frames_Cam1/" + image.split("/")[-1][-9:-4] + ".pkl")
                
                pare_pred = joblib.load(pkl_path_image)

                print(pare_pred)
                print(type(pare_pred))

                """
                if output is None or output["smpl_vertices"] is None:                
                    print("Missing detection")                
                    continue
                # return output            
                smpl_pred = Mesh(v=output["smpl_vertices"][0], f=get_smpl_faces())            
                smpl_pred.write_ply("../CHORE/CHORE_chair_all/mocap/" + imgfile)
                pred_pose = batch_rot2aa(torch.from_numpy(output["pred_pose"][0])).reshape(-1).numpy().tolist()           
                pred_shape = output["pred_shape"].reshape(-1).tolist()
                new_json = {"pose": pred_pose,"betas": pred_shape}            
                with open("../CHORE/CHORE_chair_all/mocap_param/" + imgfile[:4] + ".json", "w") as f:  
                     json.dump(new_json, f, indent=4)
                """
                #save files

                shutil.copyfile(image, final_folder_path + "/k1.color.jpg")
                cv2.imwrite(final_folder_path + "/k1.person_mask.jpg", body_mask * 225)
                cv2.imwrite(final_folder_path + "/k1.obj_mask.jpg", obj_mask * 225)

                #log_mask(img, body_mask, "body mask", {0: "background", 255: "body"})
                #log_mask(img, obj_mask, "object mask", {0: "background", 255: "object"})

                break
            break
        break
    



    """

    res = get_img_lbl()

    for imgfile in os.listdir("./CHORE_chair_all/images/"):
        try:
            print(imgfile)
            #break
            os.makedirs("./CHORE_chair_all/CHORE_format/" + imgfile[:4], exist_ok=True)

            pre, ext = os.path.splitext(imgfile)

            with open("./CHORE_chair_all/openpose/" + pre + "_keypoints.json", "r") as f:
                a = json.load(f)
            # break

            new_json = {
                "body_joints": a["people"][0]["pose_keypoints_2d"],
                "face_joints": a["people"][0]["face_keypoints_2d"],
                "left_hand_joints": a["people"][0]["hand_left_keypoints_2d"],
                "right_hand_joints": a["people"][0]["hand_right_keypoints_2d"],
            }

            with open("./CHORE_chair_all/CHORE_format/" + imgfile[:4] + "/k1.color.json", "w") as f:
                json.dump(new_json, f, indent=4)

            print("./CHORE_chair_all/mocap_param/" + imgfile[:4] + ".json")
            shutil.copyfile("./CHORE_chair_all/images/" + imgfile, "./CHORE_chair_all/CHORE_format/" + imgfile[:4] + "/k1.color.jpg")
            shutil.copyfile("./CHORE_chair_all/mocap/" + imgfile, "./CHORE_chair_all/CHORE_format/" + imgfile[:4] + "/k1.mocap.ply")
            shutil.copyfile("./CHORE_chair_all/mocap_param/" + imgfile[:4] + ".json", "./CHORE_chair_all/CHORE_format/" + imgfile[:4] + "/k1.mocap.json")
            shutil.copyfile("./CHORE_chair_all/body_masks/" + imgfile, "./CHORE_chair_all/CHORE_format/" + imgfile[:4] + "/k1.person_mask.jpg")
            shutil.copyfile("./CHORE_chair_all/obj_masks/" + imgfile, "./CHORE_chair_all/CHORE_format/" + imgfile[:4] + "/k1.obj_mask.jpg")
            
            # break
        except:
            shutil.rmtree("./CHORE_chair_all/CHORE_format/" + imgfile[:4]) 

    {
        "body_joints": a["people"][0]["pose_keypoints_2d"],
        "face_joints": a["people"][0]["face_keypoints_2d"],
        "left_hand_joints": a["people"][0]["hand_left_keypoints_2d"],
        "right_hand_joints": a["people"][0]["hand_right_keypoints_2d"],
    }

    with open("/data/huangd/log/pred_contact.pkl", "rb") as f:
        a = pickle.load(f)

    human2chore = {
        "head": [0],
        "neck": [0],
        "shoulders": [11],
        "arms": [5, 10],
        "arm": [5, 10],
        "fore arms": [4, 9],
        "upper arms": [5, 10],
        "hands": [2, 7],
        "front": [11],
        "back": [11],
        "lower body": [11, 3, 8],
        "butt": [11],
        "legs": [3, 8],
        "thighs": [3, 8],
        "calfs": [3, 8],
        "feet": [1, 6],
    }
    lbl_dir = "/data/huangd/chair_all_lbl"
    img2lbl = {}
    for lbl_file in os.listdir(lbl_dir):
        st = int(lbl_file.split("-")[1])
        print(lbl_file)
        with open("/data/huangd/chair_all_lbl/" + lbl_file, newline='') as f:
            reader = csv.reader(f)
            for idx, row in enumerate(reader):
                imglbls = []
                if row[1] != '':
                    imglbls += human2chore[row[1]]
                if row[4] != '':
                    imglbls += human2chore[row[4]]
                if row[7] != '':
                    imglbls += human2chore[row[7]]
                if row[10] != '':
                    imglbls += human2chore[row[10]]

                if len(imglbls):
                    img2lbl[st + idx] = imglbls
                    
    with open("/data/huangd/log/humanlbl_contact.pkl", "wb") as f:
        pickle.dump(img2lbl, f)
    """
