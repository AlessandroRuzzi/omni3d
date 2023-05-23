from pathlib import Path
from os.path import join
import json
import torch
import cv2
import numpy as np


def get_seg(path: str):
    path = Path(path)
    seg = path.parent.parent.stem
    if "Seg" not in seg:
        raise Exception(
            "not valid path for get_seg function. The path should be like ~/intercap_subset/01/02/Seg_1/Mesh/00001_second_obj.ply")
    else:
        return int(seg[4:])


def get_camera_file_name(frame):
    if frame == 0:
        return "Color.json"
    return f"Color_{frame + 1}.json"


def get_rotation_translation(root: str, seg: int, frame: int):
    """
    root: should be the Data folder containing the camera information: https://github.com/YinghaoHuang91/InterCap/tree/master/Data
    frame is zero_indexed
    seg is zero_indexed
    """

    seg_map = {
        0: "first",
        1: "second",
        2: "third"
    }

    calib_path = join(root, f"calibration_{seg_map[seg]}", get_camera_file_name(frame=frame))
    with open(calib_path, "r") as json_file:
        cam_info = json.load(json_file)
    r = torch.tensor(cv2.Rodrigues(np.array(cam_info['R']))[0], dtype=torch.float32)
    t = torch.tensor(cam_info['T'])
    return r, t

def get_calib_dist(root: str, seg: int, frame: int):
    """
    root: should be the Data folder containing the camera information: https://github.com/YinghaoHuang91/InterCap/tree/master/Data
    frame is zero_indexed
    seg is zero_indexed
    """

    seg_map = {
        0: "first",
        1: "second",
        2: "third"
    }

    calib_path = join(root, f"calibration_{seg_map[seg]}", get_camera_file_name(frame=frame))
    with open(calib_path, "r") as json_file:
        cam_info = json.load(json_file)
    calib = cam_info["camera_mtx"]
    dist = cam_info["k"]
    return calib, dist