from pathlib import Path
from .base_dataset import BaseDataset
from termcolor import cprint
from configs.paths import intercap_root, intercap_calib
import os
from os.path import exists
import numpy as np
import h5py
import torch
from PIL import Image
import re
import torchvision.transforms as transforms
from utils import intercap_utils as iu
from tqdm import tqdm
import cv2
from lib_smpl.smpl_utils import get_smplh


class IntercapDataset(BaseDataset):
    def initialize(self, opt, phase='train', cat=None):
        self.opt = opt
        self.phase = phase
        self.data = []

        pbar = tqdm(desc="Number of loaded images", unit="images")

        for root, _, files in os.walk(intercap_root):
            if "Mesh" not in root:
                continue
            for h5_file in files:
                if "second_obj" not in h5_file or ".h5" not in h5_file:
                    continue

                h5_path = os.path.join(root, h5_file)
                self.data.append({
                    "h5_path": h5_path,
                    "cat": "None" # TODO: find cat label for each object
                })
                pbar.update(1)     

        pbar.close()

        np.random.default_rng(seed=0).shuffle(self.data)

        if opt.max_dataset_size < len(self.data):
            self.data = self.data[:opt.max_dataset_size]
        cprint('[*] %d samples loaded.' % (len(self.data)), 'yellow')

        self.N = len(self.data)

    def __getitem__(self, index):

        cat_name = self.data[index]["cat"]
        sdf_h5_file = self.data[index]["h5_path"]

        h5_f = h5py.File(sdf_h5_file, 'r')
        sdf = h5_f['pc_sdf_sample'][:].astype(np.float32)
        sdf = torch.Tensor(sdf).view(1, 64, 64, 64).permute(0, 3, 2, 1)

        thres = self.opt.trunc_thres
        if thres != 0.0:
            sdf = torch.clamp(sdf, min=-thres, max=thres)

        ret = {
            'sdf': sdf,
            'cat_str': cat_name, #TODO
            'path': sdf_h5_file,
        }

        return ret

    def __len__(self):
        return self.N

    def name(self):
        return 'IntercapDataset'

class IntercapImgDataset(BaseDataset):
    def initialize(self, opt, phase='train', cat=None):
        self.opt = opt
        self.phase = phase
        self.data = []

        pbar = tqdm(desc="Number of loaded images", unit="images")

        for root, _, files in os.walk(intercap_root):
            if "Mesh" not in root:
                continue
            for h5_file in files:
                if "second_obj" not in h5_file or ".h5" not in h5_file:
                    continue

                h5_path = os.path.join(root, h5_file)
                kid = int(h5_path.split("_")[-2][1])
                pvqout_path = h5_path.replace("sdf.h5", "new_pvqout.npz")
                obj_path = h5_path.replace("_sdf.h5", ".obj")
                seg = iu.get_seg(obj_path)

                img_path = os.path.join(str(Path(h5_path).parent.parent), f"Frames_Cam{kid+1}", "color", f"{str(Path(h5_path).stem).split('_')[0]}.jpg")
                print(img_path)
                self.data.append({
                    "h5_path": h5_path,
                    "cat": "None", # TODO: find cat label for each object
                    "kid": kid,
                    "seg": seg,
                    "obj_path": obj_path,
                    "pvqout_path": pvqout_path,
                    "img_path": img_path
                })
                pbar.update(1)     

        pbar.close()

        np.random.default_rng(seed=0).shuffle(self.data)

        if opt.max_dataset_size < len(self.data):
            self.data = self.data[:opt.max_dataset_size]
        cprint('[*] %d samples loaded.' % (len(self.data)), 'yellow')

        self.N = len(self.data)
        self.to_tensor = transforms.ToTensor()


    def __getitem__(self, index):
        data = self.data[index]
        path = data['h5_path']
        cat_name = data["cat"]


        h5_file = h5py.File(path, 'r')

        sdf = torch.from_numpy(h5_file['pc_sdf_sample'][:].astype(np.float32))
        sdf = sdf.permute(2, 1, 0)
        code = torch.from_numpy(np.load(data['pvqout_path'])[
                                'code'].astype(np.float32))
        codeix = torch.from_numpy(np.load(data['pvqout_path'])[
                                'codeix'].astype(np.int64))

        norm_params = h5_file['norm_params'][:].astype(np.float32)
        bbox = h5_file['sdf_params'][:].astype(np.float32)
        norm_params = torch.Tensor(norm_params)
        bbox = torch.Tensor(bbox).view(2, 3)

        bbox_scale = (bbox[1, 0]-bbox[0, 0]) * norm_params[3]
        bbox_center = (bbox[0] + bbox[1]) / 2.0 * \
            norm_params[3] + norm_params[:3]
        bbox = torch.cat([bbox_center, bbox_scale.view(1)], dim=0)

        mesh_obj_path = data['obj_path']

        h5_file.close()

        thres = self.opt.trunc_thres
        if thres != 0.0:
            sdf = torch.clamp(sdf, min=-thres, max=thres)

        calibration_matrix, dist_coefs = iu.get_calib_dist(intercap_calib, data["seg"], data["kid"])

        img_path = data["img_path"]
        img = Image.open(img_path).convert('RGB')
        img = self.to_tensor(img)

        ret = {
            'sdf': sdf, 
            'z_q': code, 
            'idx': codeix, 
            'path': path,
            'img': img, 
            'img_path': img_path, 
            'mask': "TODO", #TODO
            'mask_path': "TODO", #TODO
            'cat_str': cat_name, #TODO
            'calibration_matrix': calibration_matrix,
            'dist_coefs': dist_coefs, 
            'bbox': bbox, 
            'smpl': "TODO", #TODO
            'obj_path': mesh_obj_path
        }
        return ret


    def __len__(self):
        return self.N


    def name(self):
        return 'IntercapImgDataset'