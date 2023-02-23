from dataset.dataloader import CreateDataLoader
from dataset.train_options import TrainOptions
import wandb
from tqdm import tqdm
import json
import numpy as np
import torch
from dataset import behave_camera_utils as bcu
import cv2

wandb.init(project = "Omni3D")


def plot_box_and_label(image, lw, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255), font=cv2.FONT_HERSHEY_COMPLEX):
    # Add one xyxy box to image with label
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h - 3 >= 0  # label fits outside box
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), font, lw / 3, txt_color,
                    thickness=tf, lineType=cv2.LINE_AA)

def generate_colors(i, bgr=False):
    hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
            '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
    palette = []
    for iter in hex:
        h = '#' + iter
        palette.append(tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4)))
    num = len(palette)
    color = palette[int(i) % num]
    return (color[2], color[1], color[0]) if bgr else color

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

def transform_img(img_path, bbox_corners):
    N = bbox_corners.shape[0]
    bbox_corners = torch.tensor(bbox_corners).view(N, -1, 2)
    bbox_corners = torch.reshape(bbox_corners, (N,-1,2))

    top = int(torch.min(bbox_corners[:, :, 1], dim=1)[0].int())
    left = int(torch.min(bbox_corners[:, :, 0], dim=1)[0].int())
    bottom = int(torch.max(bbox_corners[:, :, 1], dim=1)[0].int())
    right = int(torch.max(bbox_corners[:, :, 0], dim=1)[0].int())

    img = cv2.imread(img_path[0])
    xyxy = [left,top, right, bottom]
    #print((xyxy))
    plot_box_and_label(img, max(round(sum(img.shape) / 2 * 0.003), 2), xyxy, color=generate_colors(1, True))

    images = wandb.Image(img, caption="Image with projected bounding boxes")
    wandb.log({"Image YOLOv6" : images})

    return xyxy

category = [ {'id' : 0, 'name' : 'backpack', 'supercategory' : ""}, {'id' : 1, 'name' :'basketball', 'supercategory' : ""}, {'id' : 2, 'name' :'boxlarge', 'supercategory' : ""}, 
             {'id' : 3, 'name' :'boxlong', 'supercategory' : ""}, {'id' : 4, 'name' :'boxmedium', 'supercategory' : ""}, {'id' : 5, 'name' :'boxsmall', 'supercategory' : ""}, 
             {'id' : 6, 'name' :'boxtiny', 'supercategory' : ""}, {'id' : 7, 'name' :'chairblack', 'supercategory' : ""},{'id' : 8, 'name' :'chairwood', 'supercategory' : ""},
             {'id' : 9, 'name' :'keyboard', 'supercategory' : ""},{'id' : 10, 'name' :'monitor', 'supercategory' : ""}, {'id' : 11, 'name' :'plasticcontainer', 'supercategory' : ""}, 
             {'id' : 12, 'name' :'stool', 'supercategory' : ""}, {'id' : 13, 'name' :'suitcase', 'supercategory' : ""}, {'id' : 14, 'name' :'tablesmall', 'supercategory' : ""}, 
             {'id' : 15, 'name' :'tablesquare', 'supercategory' : ""}, {'id' : 16, 'name' :'toolbox', 'supercategory' : ""}, {'id' : 17, 'name' :'trashbin', 'supercategory' : ""}, 
             {'id' : 18, 'name' :'yogaball', 'supercategory' : ""}, {'id' : 19, 'name' :'yogamat', 'supercategory' : ""}, {'id' : 20, 'name' :'person', 'supercategory' : ""}]

opt = TrainOptions().parse()
train_dl, val_dl, test_dl = CreateDataLoader(opt)
train_ds, test_ds = train_dl.dataset, test_dl.dataset

val_ds = val_dl.dataset if val_dl is not None else None

for id_data,dl in enumerate([(train_dl,"Train")]):
#for id_data,dl in enumerate([(test_dl,"Test")]):
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
                            "file_path"		  : data["img_path"][0],
                            "K"			      : data['calibration_matrix'].detach().cpu().numpy().reshape(3,3).tolist() ,
                            "src_90_rotate"	  : 0,			
                            "src_flagged"	  : False,	

                    })

        for j,elem in enumerate(category):
            if elem['name'] == data["cat_str"][0]:
                    pos_category = j
                    break

        calibration_matrix = data['calibration_matrix'].cpu().numpy()
        dist_coefs = data['dist_coefs'].cpu().numpy()
        projector = [
        bcu.get_local_projector(c, d) for c, d in zip(calibration_matrix, dist_coefs)
         ]

        bbox_project = data['bbox'].cuda()
        bbox_project[:, :2] = bbox_project[:, :2] * -1
        patch_coord_projected, bbox_corners = calc_patch_coord(bbox_project, projector)
        bbox2d = transform_img(data["img_path"], bbox_corners)

        bbox = data['bbox'].detach().cpu().numpy()
        obj_length = float(bbox[0,3])


        ones = -1 * np.ones(24).reshape(8,3)

        print(bbox_project)
        print(obj_length)

        object.append({

                            "id"			  : i * 2,					
                            "image_id"		  : i,	
                            "dataset_id"	  : id_data,				
                            "category_id"	  : category[pos_category]['id'],					
                            "category_name"	  : category[pos_category]['name'],		
                            
                            "valid3D"		  : True,				   
                            "bbox2D_tight"	  : [-1,-1,-1,-1],		
                            "bbox2D_proj"	  : bbox2d,			
                            "bbox2D_trunc"	  : [-1,-1,-1,-1],			
                            "bbox3D_cam"	  : ones.tolist(),
                            #"center_cam"	  : bbox[0,:3].tolist(),		
                            "center_cam"	  : bbox_project[0,:3].detach().cpu().numpy().tolist(),			
                            "dimensions"	  : [obj_length, obj_length, obj_length],
                            "R_cam"		      : np.eye(3).tolist(),	

                            "behind_camera"	  : False,				
                            "visibility"	  : -1, 		
                            "truncation"	  : -1, 				
                            "segmentation_pts": -1, 					
                            "lidar_pts" 	  : -1, 					
                            "depth_error"	  : -1,				
       
                    })

        calibration_matrix = data['calibration_matrix'].cpu().numpy()
        dist_coefs = data['dist_coefs'].cpu().numpy()
        projector = [
        bcu.get_local_projector(c, d) for c, d in zip(calibration_matrix, dist_coefs)
         ]
        
        verts = data['body_mesh_verts']
        human_center = [(torch.min(verts[:,0]) + (torch.max(verts[:,0]) - torch.min(verts[:,0])) / 2.0).detach().cpu().numpy(), 
                (torch.min(verts[:,1]) + (torch.max(verts[:,1]) - torch.min(verts[:,1])) / 2.0).detach().cpu().numpy(),
                (torch.min(verts[:,2]) + (torch.max(verts[:,2]) - torch.min(verts[:,2])) / 2.0).detach().cpu().numpy()]
        
        bbox_project = human_center
        obj_length = max((torch.max(verts[:,0]) - torch.min(verts[:,0])).detach().cpu().numpy(), (torch.max(verts[:,1]) - torch.min(verts[:,1])).detach().cpu().numpy(),
                          (torch.max(verts[:,2]) - torch.min(verts[:,2])).detach().cpu().numpy())
        bbox_to_project = human_center.copy()
        bbox_to_project.append(obj_length)
        bbox_to_project = [bbox_to_project]
        bbox_to_project = torch.FloatTensor(np.array(bbox_to_project)).cuda()
        #bbox_project[:, :2] = bbox_project[:, :2] * -1

        print(bbox_project)
        print(bbox_to_project)
        print(bbox_to_project.shape)
        print(obj_length)

        patch_coord_projected, bbox_corners = calc_patch_coord(bbox_to_project, projector)
        bbox2d = transform_img(data["img_path"], bbox_corners)

        

        

        print("-----------------------------")
        

        object.append({

                            "id"			  : (i * 2)+1,					
                            "image_id"		  : i,	
                            "dataset_id"	  : id_data,				
                            "category_id"	  : category[pos_category]['id'],					
                            "category_name"	  : category[pos_category]['name'],		
                            
                            "valid3D"		  : True,				   
                            "bbox2D_tight"	  : [-1,-1,-1,-1],		
                            "bbox2D_proj"	  : bbox2d,			
                            "bbox2D_trunc"	  : [-1,-1,-1,-1],			
                            "bbox3D_cam"	  : ones.tolist(),
                            #"center_cam"	  : bbox[0,:3].tolist(),		
                            "center_cam"	  : bbox_project.tolist(),			
                            "dimensions"	  : [obj_length, obj_length, obj_length],
                            "R_cam"		      : np.eye(3).tolist(),	

                            "behind_camera"	  : False,				
                            "visibility"	  : -1, 		
                            "truncation"	  : -1, 				
                            "segmentation_pts": -1, 					
                            "lidar_pts" 	  : -1, 					
                            "depth_error"	  : -1,				
       
                    })
        if i == 2:
            break

    dataset['info'] = info
    dataset['images'] = image
    dataset['categories'] = category
    dataset['annotations'] = object

    out_file = open(f'/data/aruzzi/Behave/Behave_person_{dl[1]}.json',"w") 
    json.dump(dataset, out_file, indent = 2)
    out_file.close()
