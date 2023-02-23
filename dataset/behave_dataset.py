from pathlib import Path
from .base_dataset import BaseDataset
from termcolor import cprint
from configs.paths import behave_seqs_path, behave_calibs_path
import os
from os.path import exists
import numpy as np
import h5py
import torch
from PIL import Image
import re
import torchvision.transforms as transforms
from dataset import behave_camera_utils as bcu
from tqdm import tqdm
from pytorch3d.io import load_ply
import cv2

split = {
    "train": {"Date01", "Date02", "Date05", "Date06", "Date07"},
    "val": {"Date04"},
    "test": {"Date03"}
}

class BehaveDataset(BaseDataset):

    def initialize(self, opt, phase='train', cat=None):
        if phase in split:
            self.dates = split[phase]
        else:
            assert False, f"Unknown phase for BEHAVE dataset: {phase}"

        cprint(f"[*] Loading BEHAVE dataset for phase {phase}", 'yellow')
        if cat is not None:
            cprint(f"Using category {cat}", "yellow")

        self.opt = opt
        self.phase = phase
        self.cats_list = []
        self.sdf_list = []

        pbar = tqdm(desc="Number of loaded images", unit="images")
        seq_list = [seq for seq in os.listdir(behave_seqs_path) if seq.split("_")[0] in self.dates]

        for seq in seq_list:
            for root, _, files in os.walk(os.path.join(behave_seqs_path, seq)):
                if 'fit' not in root:
                    continue
                for f in files:
                    if re.match(r".*_fit_k\d_sdf.h5", f):
                        self.sdf_list.append(os.path.join(root,f))
                        self.cats_list.append(f.split("_")[0])
                        pbar.update(1)
                    if len(self.sdf_list) >= opt.max_dataset_size:
                        break
                if len(self.sdf_list) >= opt.max_dataset_size:
                        break
            if len(self.sdf_list) >= opt.max_dataset_size:
                        break
        pbar.close()

        np.random.default_rng(seed=0).shuffle(self.sdf_list)
        np.random.default_rng(seed=0).shuffle(self.cats_list)

        # need to check the seed for reproducibility
        cprint('[*] %d samples loaded.' % (len(self.sdf_list)), 'yellow')

        self.N = len(self.sdf_list)

    def __getitem__(self, index):
        cat_name = self.cats_list[index]
        sdf_h5_file = self.sdf_list[index]

        h5_f = h5py.File(sdf_h5_file, 'r')
        sdf = h5_f['pc_sdf_sample'][:].astype(np.float32)
        sdf = torch.Tensor(sdf).view(1, 64, 64, 64)

        thres = self.opt.trunc_thres
        if thres != 0.0:
            sdf = torch.clamp(sdf, min=-thres, max=thres)

        ret = {
            'sdf': sdf,
            'cat_str': cat_name,
            'path': sdf_h5_file,
        }

        return ret

    def __len__(self):
        return self.N

    def name(self):
        return 'BehaveDataset'

class BehaveImgDataset(BaseDataset):
    def initialize(self, opt, phase='train', cat=None):
        if phase in split:
            self.dates = split[phase]
        elif phase == "test-chore":
            self.dates = split["test"]
        else:
            assert False, f"Unknown phase for BEHAVE dataset: {phase}"

        cprint(f"[*] Loading BEHAVE dataset for phase {phase}", 'yellow')
        if cat is not None:
            cprint(f"Using category {cat}", "yellow")

        self.opt = opt
        self.phase = phase
        self.data = []

        self.intrinsics = [bcu.load_intrinsics(os.path.join(behave_calibs_path, "intrinsics"), i) for i in range(4)]

        seq_list = [seq for seq in os.listdir(behave_seqs_path) if seq.split("_")[0] in self.dates]

        self.camera_params = {}
        for i in range(1, 8):
            self.camera_params[f"Date0{i}"] = bcu.load_kinect_poses_back(
                os.path.join(behave_calibs_path, f"Date0{i}", "config"),
                [0,1,2,3],
                True
            )

        pbar = tqdm(desc="Number of loaded images", unit="images")
        for seq in seq_list:
            for root, _, files in os.walk(os.path.join(behave_seqs_path, seq)):
                if "fit" not in root:
                    continue
                for f in files:
                    if re.match(r".*_fit_k\d_sdf.h5", f):
                        
                        kid = int(f.split("_")[-2][1])
                        category = f.split("_")[0]
                        
                        h5_path = os.path.join(root,f)

                        pvqout_path = h5_path.replace("sdf.h5", "pvqout.npz")
                        obj_path = h5_path.replace("_sdf.h5", ".obj")

                        par = str(Path(h5_path).parents[2])
                        
                        img_path = os.path.join(par, f"k{kid}.color.jpg")
                        mask_path = os.path.join(par, f"k{kid}.obj_rend_mask.jpg")
                        smpl_path = os.path.join(par, 'person', 'fit02', 'person_fit.pkl')

                        # filter out images with high occlusion for test-chore
                        if phase == "test-chore":
                            full_mask_path = os.path.join(par, f"k{kid}.obj_rend_full.jpg")
                            
                            if not os.path.exists(full_mask_path):
                                cprint(
                                    """Full mask not found, please download the full masks from the link below:
                                    https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/behave-test-object-fullmask.zip
                                    Take into account that if this link doesn't work it's probably moved, please check the CHORE github repo if this happens:
                                    https://github.com/xiexh20/CHORE
                                    """,
                                    "red")
                            
                            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127
                            full_mask = cv2.imread(full_mask_path, cv2.IMREAD_GRAYSCALE) > 127
                            if np.sum(mask) / np.sum(full_mask) < 0.3:
                                continue
                        
                        if exists(img_path) and exists(mask_path) and exists(pvqout_path) and exists(obj_path):
                            self.data.append({
                                'img_path': img_path,
                                'mask_path': mask_path,
                                'pvqout_path': pvqout_path,
                                'obj_path': obj_path,
                                'h5_path': h5_path,
                                'category': category,
                                'kid': kid,
                                'smpl_path': smpl_path,
                            })
                            pbar.update(1)
                        
                    if len(self.data) >= opt.max_dataset_size:
                        break
                if len(self.data) >= opt.max_dataset_size:
                        break
            if len(self.data) >= opt.max_dataset_size:
                        break
        pbar.close()
        
        np.random.default_rng(seed=0).shuffle(self.data)

        cprint('[*] %d data loaded.' % (len(self.data)), 'yellow')

        self.N = len(self.data)
        self.to_tensor = transforms.ToTensor()


    def __getitem__(self, index):

        data = self.data[index]
        
        path = data['h5_path']
        
        h5_file = h5py.File(path, 'r')
        
        sdf = torch.from_numpy(h5_file['pc_sdf_sample'][:].astype(np.float32))
        code = torch.from_numpy(np.load(data['pvqout_path'])['code'].astype(np.float32))
        codeix = torch.from_numpy(np.load(data['pvqout_path'])['codeix'].astype(np.int64))
        
        
        img_path = data['img_path']
        img = Image.open(img_path).convert('RGB')
        img = self.to_tensor(img)
        
        mask_path = data['mask_path']
        mask = Image.open(mask_path).convert('L')
        mask = self.to_tensor(mask)
        
        category = data['category']
        
        norm_params = h5_file['norm_params'][:].astype(np.float32)
        bbox = h5_file['sdf_params'][:].astype(np.float32)
        norm_params = torch.Tensor(norm_params)
        bbox = torch.Tensor(bbox).view(2, 3)
        
        bbox_scale = (bbox[1, 0]-bbox[0, 0]) * norm_params[3]
        bbox_center = (bbox[0] + bbox[1]) / 2.0 * norm_params[3] + norm_params[:3]
        bbox = torch.cat([bbox_center, bbox_scale.view(1)], dim=0)

        calibration_matrix = self.intrinsics[data['kid']][0]
        dist_coefs = self.intrinsics[data['kid']][1]

        mesh_obj_path = data['obj_path']

        h5_file.close()

        ret = {
            'sdf': sdf, 'z_q': code, 'idx': codeix, 'path': path,
            'img': img, 'img_path': img_path, 'mask': mask, 'mask_path': mask_path,
            'cat_str': category, 'calibration_matrix': calibration_matrix,
            'dist_coefs': dist_coefs, 'bbox': bbox, 'smpl_path': data['smpl_path'], #'human_joints': 0,
            'obj_path': mesh_obj_path
        }
        rt = (
            self.camera_params[data['date']][0][data['kid']],
            self.camera_params[data['date']][1][data['kid']],
        )
        behave_verts, faces_idx = load_ply(data['body_mesh'])
        behave_verts = behave_verts.reshape(-1, 3).numpy()
        behave_verts = bcu.global2local(behave_verts, rt[0], rt[1])
            
        behave_verts[:, :2] *= -1
        # print(behave_verts.shape, faces_idx.shape)
        # theMesh = Meshes(verts=[torch.from_numpy(behave_verts).float()], faces=faces_idx)
        ret['body_mesh_verts'] = torch.from_numpy(behave_verts)
        return ret

    def __len__(self):
        return self.N

    def name(self):
        return 'BehaveImageDataset'