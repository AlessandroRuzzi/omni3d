from pathlib import Path
from dataset.base_dataset import BaseDataset
from termcolor import cprint
from configs.paths import intercap_root, intercap_calib, smplx_model_root
import os
from os.path import exists
import numpy as np
import h5py
import torch
from PIL import Image
import re
import torchvision.transforms as transforms
from dataset import intercap_utils as iu
from tqdm import tqdm
import cv2
from lib_smpl.smpl_utils import get_smplh
import smplx
import pickle

split = {
    "train": {"01", "02", "03", "04", "05", "06", "07"},
    "val": {"08"},
    "test": {"09", "10"}
}
cats = {
    "01": "suitcase",
    "02": "skateboard",
    "03": "sprotball",
    "04": "umbrella",
    "05": "tennisracket",
    "06": "handbag",
    "07": "chair",
    "08": "bottle",
    "09": "cup",
    "10": "couch"
}

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
                sp = h5_path.split("/")[-5]
                if sp not in split[phase]:
                    continue
                cat_id = h5_path.split("/")[-4]
                cat_name = cats[cat_id]
                self.data.append({
                    "h5_path": h5_path,
                    "cat": cat_name,
                    "cat_id": cat_id,
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
        cat_id = self.data[index]["cat_id"]
        sdf_h5_file = self.data[index]["h5_path"]

        h5_f = h5py.File(sdf_h5_file, 'r')
        sdf = h5_f['pc_sdf_sample'][:].astype(np.float32)
        sdf = torch.Tensor(sdf).view(1, 64, 64, 64).permute(0, 3, 2, 1)

        thres = self.opt.trunc_thres
        if thres != 0.0:
            sdf = torch.clamp(sdf, min=-thres, max=thres)

        ret = {
            'sdf': sdf,
            'cat_str': cat_name,
            'cat_id': cat_id,
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
                sp = h5_path.split("/")[-5]
                if sp not in split[phase]:
                    continue
                kid = int(h5_path.split("_")[-2][1])
                pvqout_path = h5_path.replace("sdf.h5", "new_pvqout.npz")
                obj_path = h5_path.replace("_sdf.h5", ".obj")
                seg = iu.get_seg(obj_path)
                cat_id = h5_path.split("/")[-4]
                cat_name = cats[cat_id]
                
                img_path = os.path.join(str(Path(h5_path).parent.parent), f"Frames_Cam{kid+1}", "color", f"{str(Path(h5_path).stem).split('_')[0]}.jpg")
                img_path = img_path.replace("Res", "RGBD_Images")
                smpl_path = os.path.join(str(Path(h5_path).parent.parent), "res_2.pkl")
                
                #print(img_path)
                self.data.append({
                    "h5_path": h5_path,
                    "cat_id": cat_id,
                    "cat": cat_name,
                    "kid": kid,
                    "seg": seg,
                    "obj_path": obj_path,
                    "pvqout_path": pvqout_path,
                    "img_path": img_path,
                    "smpl_path": smpl_path,
                    "frame_number": int(str(Path(h5_path).stem).split('_')[0])
                })
                pbar.update(1)    

                if len(self.data) >= opt.max_dataset_size:
                    break 
            if len(self.data) >= opt.max_dataset_size:
                break 
        
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
        cat_id = data["cat_id"]
        
        
        h5_file = h5py.File(path, 'r')
        
        #sdf = torch.from_numpy(h5_file['pc_sdf_sample'][:].astype(np.float32))
        #sdf = sdf.permute(2, 1, 0)
        #code = torch.from_numpy(np.load(data['pvqout_path'])[
        #                        'code'].astype(np.float32))
        #codeix = torch.from_numpy(np.load(data['pvqout_path'])[
        #                          'codeix'].astype(np.int64))
        
        norm_params = h5_file['norm_params'][:].astype(np.float32)
        #bbox = h5_file['sdf_params'][:].astype(np.float32)
        bbox = np.array([-1.1672062, -1.134178,  -1.1588331,  1.1859713,  1.2189994,  1.1943444])
        norm_params = torch.Tensor(norm_params)
        bbox = torch.Tensor(bbox).view(2, 3)
        
        bbox_scale = (bbox[1, 0]-bbox[0, 0]) * norm_params[3]
        bbox_center = (bbox[0] + bbox[1]) / 2.0 * \
            norm_params[3] + norm_params[:3]
        bbox = torch.cat([bbox_center, bbox_scale.view(1)], dim=0)
        
        mesh_obj_path = data['obj_path']

        h5_file.close()
        
        thres = self.opt.trunc_thres
        #if thres != 0.0:
            #sdf = torch.clamp(sdf, min=-thres, max=thres)
            
        calibration_matrix, dist_coefs = iu.get_calib_dist(intercap_calib, data["seg"], data["kid"])
        
        img_path = data["img_path"]
        img = Image.open(img_path).convert('RGB')
        img = self.to_tensor(img)
            
        ret = {
            #'sdf': sdf, 
            #'z_q': code, 
            #'idx': codeix, 
            'path': path,
            'img': img, 
            'img_path': img_path, 
            'mask': "TODO", #TODO
            'mask_path': "TODO", #TODO
            'cat_str': cat_name,
            'cat_id': cat_id,
            'calibration_matrix': calibration_matrix,
            'dist_coefs': dist_coefs, 
            'bbox': bbox, 
            'body_mesh_verts': load_smpl(data["smpl_path"], data["seg"],  data["frame_number"], data["kid"]),
            'obj_path': mesh_obj_path
        }
        return ret
    
    def __len__(self):
        return self.N


    def name(self):
        return 'IntercapImgDataset'

def load_smpl(smpl_paths, seg, frame, kid):
    with open(smpl_paths, 'rb') as fin:
        data = pickle.load(fin)
    GENDER = 'female'
    
    params = {'body_pose': data['body_pose'].reshape([-1, 63])[frame, :].reshape(-1,63), 
        'right_hand_pose':data['right_hand_pose'][frame][np.newaxis, ...], 
        'betas': data['betas'][:, :10][frame][np.newaxis, ...], 
        'transl':data['transl'][frame][np.newaxis, ...], 
        'global_orient':data['global_orient'][frame][np.newaxis, ...], 
        'expression': data['expression'][frame][np.newaxis, ...]}
    
    def params2torch(params, dtype = torch.float32):
        return {k: torch.from_numpy(np.copy(v)).type(dtype) for k, v in params.items()}

    model = smplx.create(model_path=smplx_model_root,
                        model_type='smplx',
                        is_rhand=False,
                        gender=GENDER,
                        num_pca_comps=12,
                        flat_hand_mean = True,
                        use_pca=True,
                        batch_size=1)

    lh_parms = params2torch(params)
    output = model(**lh_parms)
    
    jtr = output.vertices[0,:,:]
    # select based on the names in https://github.com/vchoutas/smplx/blob/main/smplx/joint_names.py
    #jtr = torch.cat((jtr[:, :22], jtr[:, 28:29], jtr[:, 43:44]), dim=1) # (1, 24, 3)
    #jtr = jtr[0].detach().numpy()
    
    r, t = iu.get_rotation_translation(intercap_calib, seg, kid)
    jtr = (r@jtr.T ).T + t
    
    return jtr