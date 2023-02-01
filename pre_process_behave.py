from dataset.dataloader import CreateDataLoader
from dataset.train_options import TrainOptions
import wandb
from tqdm import tqdm
import json
import numpy as np
import torch
from dataset import behave_camera_utils as bcu

wandb.init(project = "Omni3D")

def __calc_patch_coord(bbox_center, projector, nP, l):
    N = bbox_center.shape[0]
    res = []
    for i in range(N):
        """
        Match patch coordinates with patches from the extracted code by rotating them. 
        Before rotation, Entry i, j, k corresponds to the z, -y, -x axis.
        """
        lspace = torch.linspace(-l[i], l[i], nP)
        x = lspace.view(1, -1, 1, 1, 1).repeat(1, 1, nP, nP, 1)
        y = lspace.view(1, 1, -1, 1, 1).repeat(1, nP, 1, nP, 1)
        z = lspace.view(1, 1, 1, -1, 1).repeat(1, nP, nP, 1, 1)
        xyz = torch.cat([x, y, z], dim=-1).cuda()  # [1, nP, nP, nP, 3]
        xyz += bbox_center[i].view(1, 1, 1, 1, 3)
        xyz = torch.flip(xyz, dims=(1,2)).permute(0, 3, 2, 1, 4)
        xyz = xyz.reshape(-1, 3).cpu().numpy()

        # At this point, xyz is a tensor containing the coordinates of the center of patches
        # Then we need to project them to the image plane
        res.append(projector[i](xyz))

        return torch.from_numpy(np.array(res)).view(N, nP, nP, nP, 2).detach().cpu().numpy()

def calc_patch_coord(bbox, projector):
    bbox_center = bbox[:, :3]
    bbox_len = bbox[:, 3]
    
    nP = 8
    l = bbox_len * (nP - 1) / nP / 2
    patch_coord_projected = __calc_patch_coord(bbox_center, projector, nP, l)

    l = bbox_len / 2
    bbox_corners = __calc_patch_coord(bbox_center, projector, 2, l)

    return patch_coord_projected, bbox_corners

category = [ {'id' : 0, 'name' : 'backpack', 'supercategory' : ""}, {'id' : 1, 'name' :'basketball', 'supercategory' : ""}, {'id' : 2, 'name' :'boxlarge', 'supercategory' : ""}, 
             {'id' : 3, 'name' :'boxlong', 'supercategory' : ""}, {'id' : 4, 'name' :'boxmedium', 'supercategory' : ""}, {'id' : 5, 'name' :'boxsmall', 'supercategory' : ""}, 
             {'id' : 6, 'name' :'boxtiny', 'supercategory' : ""}, {'id' : 7, 'name' :'chairblack', 'supercategory' : ""},{'id' : 8, 'name' :'chairwood', 'supercategory' : ""},
             {'id' : 9, 'name' :'keyboard', 'supercategory' : ""},{'id' : 10, 'name' :'monitor', 'supercategory' : ""}, {'id' : 11, 'name' :'plasticcontainer', 'supercategory' : ""}, 
             {'id' : 12, 'name' :'stool', 'supercategory' : ""}, {'id' : 13, 'name' :'suitcase', 'supercategory' : ""}, {'id' : 14, 'name' :'tablesmall', 'supercategory' : ""}, 
             {'id' : 15, 'name' :'tablesquare', 'supercategory' : ""}, {'id' : 16, 'name' :'toolbox', 'supercategory' : ""}, {'id' : 17, 'name' :'trashbin', 'supercategory' : ""}, 
             {'id' : 18, 'name' :'yogaball', 'supercategory' : ""}, {'id' : 19, 'name' :'yogamat', 'supercategory' : ""}]

opt = TrainOptions().parse()
train_dl, val_dl, test_dl = CreateDataLoader(opt)
train_ds, test_ds = train_dl.dataset, test_dl.dataset

val_ds = val_dl.dataset if val_dl is not None else None

for id_data,dl in enumerate([(train_dl,"Train"), (val_dl,"Validation"), (test_dl,"Test")]):
    dataset = {}
    info = {

            "id"			: id_data,
            "source"		: "Behave",
            "name"			: f'Behave {dl[1]}',
            "split"			: f"{dl[1]}",
            "version"		: "1.0",
            "url"			: "",

            }
    image = []
    object = []

    for i, data in tqdm(enumerate(dl[0]), total=len(dl[0])):
        pos_category = -1
        image.append({

                        	"id"			  : i,
                            "dataset_id"	  : id_data,
                            "width"			  : 2048,
                            "height"		  : 1536,
                            "file_path"		  : data["img_path"],
                            "K"			      : data['calibration_matrix'].detach().cpu().numpy().tolist() ,
                            "src_90_rotate"	  : 0,			
                            "src_flagged"	  : False,	

                    })

        for j,elem in enumerate(category):
            if elem['name'] == data["cat_str"]:
                    pos_category = j

        calibration_matrix = data['calibration_matrix'].cpu().numpy()
        dist_coefs = data['dist_coefs'].cpu().numpy()
        projector = [
        bcu.get_local_projector(c, d) for c, d in zip(calibration_matrix, dist_coefs)
         ]

        patch_coord_projected, bbox_corners = calc_patch_coord(data['bbox'].cuda(), projector)

        bbox = data['bbox'].detach().cpu().numpy()

        print(bbox,patch_coord_projected.shape, bbox_corners.shape)

        """
        object.append({

                            "id"			  : i,					
                            "image_id"		  : i,					
                            "category_id"	  : category[pos_category]['id'],					
                            "category_name"	  : category[pos_category]['name'],		
                            
                            "valid3D"		  : True,				   
                            "bbox2D_tight"	  : [-1,-1,-1,-1],		
                            "bbox2D_proj"	  : float(patch_coord_projected).tolist(),			# 2D corners projected from bbox3D
                            "bbox2D_trunc"	  : [],			# 2D corners projected from bbox3D then truncated
                            "bbox3D_cam"	  : bbox_corners.tolist(),
                            "center_cam"	  : bbox[0,:3].tolist(),				
                            "dimensions"	  : [bbox[0,3], bbox[0,3], bbox[0,3]],
                            "R_cam"		      : np.eye(3).tolist(),	

                            "behind_camera"	  : -1,				
                            "visibility"	  : -1, 		
                            "truncation"	  : -1, 				
                            "segmentation_pts": -1, 					
                            "lidar_pts" 	  : -1, 					
                            "depth_error"	  : -1,				
       
                    })
        """
        break

    dataset['info'] = info
    dataset['image'] = image
    dataset['cateogory'] = category
    dataset['object'] = object

    out_file = open(f'/data/aruzzi/Behave/Behave_{dl[1]}.json',"w") 
    json.dump(dataset, out_file, indent = 2)
    out_file.close()
