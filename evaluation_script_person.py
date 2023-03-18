import json
from operator import itemgetter
import math
import torch
import wandb
import os
from glob import glob
from pytorch3d.ops import box3d_overlap
import numpy as np
import cv2
from cubercnn.vis import draw_3d_box, draw_text, draw_scene_view
import wandb
from cubercnn import util

wandb.init(project = "Omni3D")

category = [ {'id' : 0, 'name' : 'backpack', 'supercategory' : ""}, {'id' : 1, 'name' :'basketball', 'supercategory' : ""}, {'id' : 2, 'name' :'boxlarge', 'supercategory' : ""}, 
             {'id' : 3, 'name' :'boxlong', 'supercategory' : ""}, {'id' : 4, 'name' :'boxmedium', 'supercategory' : ""}, {'id' : 5, 'name' :'boxsmall', 'supercategory' : ""}, 
             {'id' : 6, 'name' :'boxtiny', 'supercategory' : ""}, {'id' : 7, 'name' :'chairblack', 'supercategory' : ""},{'id' : 8, 'name' :'chairwood', 'supercategory' : ""},
             {'id' : 9, 'name' :'keyboard', 'supercategory' : ""},{'id' : 10, 'name' :'monitor', 'supercategory' : ""}, {'id' : 11, 'name' :'plasticcontainer', 'supercategory' : ""}, 
             {'id' : 12, 'name' :'stool', 'supercategory' : ""}, {'id' : 13, 'name' :'suitcase', 'supercategory' : ""}, {'id' : 14, 'name' :'tablesmall', 'supercategory' : ""}, 
             {'id' : 15, 'name' :'tablesquare', 'supercategory' : ""}, {'id' : 16, 'name' :'toolbox', 'supercategory' : ""}, {'id' : 17, 'name' :'trashbin', 'supercategory' : ""}, 
             {'id' : 18, 'name' :'yogaball', 'supercategory' : ""}, {'id' : 19, 'name' :'yogamat', 'supercategory' : ""}, {'id' : 20, 'name' :'person', 'supercategory' : ""},
             {'id' : 21, 'name' :'interaction', 'supercategory' : ""}]


def log_bboxes(img, object_box, object_dim, object_orientation, object_cat, object_score, human_box, human_dim, human_orientation):
        
        id_cat = None
        for j,elem in enumerate(category):
            if elem['name'] == object_cat:
                id_cat = j
                break

        image_shape = img.shape[:2]  # h, w

        h, w = image_shape
        f_ndc = 1
        f = f_ndc * h / 2

        K = np.array([
            [f, 0.0, w/2], 
            [0.0, f, h/2], 
            [0.0, 0.0, 1.0]
        ])
        color = util.get_color(id_cat)

        meshes = []
        meshes_text = []
        bbox3D = object_box + object_dim

        meshes_text.append('{} {:.2f}'.format(object_cat, object_score))
        color = [c/255.0 for c in util.get_color(id_cat)]
        object_orientation = np.eye(3).tolist()
        box_mesh = util.mesh_cuboid(bbox3D, object_orientation, color=color)
        meshes.append(box_mesh)

        im_drawn_rgb, im_topdown, _ = draw_scene_view(img, K, meshes, text=meshes_text, scale=img.shape[0], blend_weight=0.5, blend_weight_overlay=0.85)

        #x3d, y3d, z3d, w3d, h3d, l3d, ry3d = object_box[0], object_box[1], object_box[2], object_dim, object_dim, object_dim, object_orientation
        #x3d, y3d, z3d = (K_inv @ (z3d*cen_2d))
        #draw_3d_box(img, K, [x3d, y3d, z3d, w3d, h3d, l3d], ry3d, color=color, thickness=int(np.round(3*img.shape[0]/500)), draw_back=True, draw_top=True)
        #draw_text(im, '{}, z={:.1f}, s={:.2f}'.format(cat, z3d, score), [x1, y1, w, h], scale=0.50*im.shape[0]/500, bg_color=color)

        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #images = wandb.Image(img, caption="Image with predicted 3D bounding boxes")
        #wandb.log({"BBOX Detected" : images})

        #img = cv2.cvtColor(im_drawn_rgb, cv2.COLOR_BGR2RGB)
        images = wandb.Image(im_drawn_rgb, caption="Frontal image with predicted 3D bounding boxes")
        wandb.log({"Frontal image" : images})

        #img = cv2.cvtColor(im_topdown, cv2.COLOR_BGR2RGB)
        images = wandb.Image(im_topdown, caption="Topdown image with predicted 3D bounding boxes")
        wandb.log({"Topdown image" : images})


def calc_num_wrong_bbox(results):
    num_wrong = 0

    for day in results:
        pred_dict = results[day]
        gt_box = pred_dict["gt_bbox_center"]
        pred_box = pred_dict["pred_bbox_center"]
        gt_length = pred_dict["gt_bbox_size"][0]
        pred_length = pred_dict["pred_bbox_size"][0]

        if not(pred_box[0] >= gt_box[0] - gt_length/2 and pred_box[0] <= gt_box[0] + gt_length/2 and pred_box[1] >= gt_box[1] - gt_length/2 and pred_box[1] <= gt_box[1] + gt_length/2 and 
                pred_box[2] >= gt_box[2] - gt_length/2 and pred_box[2] <= gt_box[2] + gt_length/2): 
            num_wrong += 1
        
    print("Number of wrong bbox: ", num_wrong)

def calc_errors_on_correct_bbox(results):
    error_dict = {'x' : 0, 'y' : 0, 'z': 0, 'l': 0 , 'num_imgs' : 0}

    for day in results:
        pred_dict = results[day]
        gt_box = pred_dict["gt_bbox_center"]
        pred_box = pred_dict["pred_bbox_center"]
        gt_length = pred_dict["gt_bbox_size"][0]
        pred_length = pred_dict["pred_bbox_size"][0]

        if pred_box[0] >= gt_box[0] - gt_length/2 and pred_box[0] <= gt_box[0] + gt_length/2 and pred_box[1] >= gt_box[1] - gt_length/2 and pred_box[1] <= gt_box[1] + gt_length/2 and pred_box[2] >= gt_box[2] - gt_length/2 and pred_box[2] <= gt_box[2] + gt_length/2: 

                error_dict['x'] += (abs((abs(pred_box[0]-gt_box[0]))/gt_length)) * 100.0
                error_dict['y'] += (abs((abs(pred_box[1]-gt_box[1]))/gt_length)) * 100.0
                error_dict['z'] += (abs((abs(pred_box[2]-gt_box[2]))/gt_length)) * 100.0
                error_dict['l'] += (abs((abs(pred_length - gt_length))/gt_length)) * 100.0
                error_dict['num_imgs'] += 1

    print("-------------------------------------")
    print("X Error: ", error_dict['x'] / error_dict['num_imgs'])
    print("Y Error: ", error_dict['y'] / error_dict['num_imgs'])
    print("Z Error: ", error_dict['z'] / error_dict['num_imgs'])
    print("Lenght Error: ", error_dict['l'] / error_dict['num_imgs'])
    print("-------------------------------------\n")

def calc_errors_using_closest_bbox(results, results_all):
    error_dict = {'x' : 0, 'y' : 0, 'z': 0, 'l': 0 , 'num_imgs' : 0}

    for day in results:
        pred_dict = results[day]
        pred_all = results_all[day]
        gt_box = pred_dict["gt_bbox_center"]
        gt_length = pred_dict["gt_bbox_size"][0]

        object_dist_list = []
        for i, bbox in enumerate(pred_all["bbox_center"]):
            object_dist_list.append(math.dist(gt_box, bbox))


        pos, element = min(enumerate(object_dist_list), key=itemgetter(1))
        #print("human distance: ",math.dist(pred_all["bbox_center"][pos], gt_box), " Confidence: ", (1-pred_all["bbox_score"][pos]))
        pred_box = pred_all["bbox_center"][pos]
        pred_length = pred_all["bbox_size"][pos][0]
    
        error_dict['x'] += (abs((abs(pred_box[0]-gt_box[0]))/gt_length)) * 100.0
        error_dict['y'] += (abs((abs(pred_box[1]-gt_box[1]))/gt_length)) * 100.0
        error_dict['z'] += (abs((abs(pred_box[2]-gt_box[2]))/gt_length)) * 100.0
        error_dict['l'] += (abs((abs(pred_length - gt_length))/gt_length)) * 100.0
        error_dict['num_imgs'] += 1

    print("-------------------------------------")
    print("X Error: ", error_dict['x'] / error_dict['num_imgs'])
    print("Y Error: ", error_dict['y'] / error_dict['num_imgs'])
    print("Z Error: ", error_dict['z'] / error_dict['num_imgs'])
    print("Lenght Error: ", error_dict['l'] / error_dict['num_imgs'])
    print("-------------------------------------\n")

def calc_errors_on_high_prob_bbox(results):
    error_dict = {'x' : 0, 'y' : 0, 'z': 0, 'l': 0 , 'num_imgs' : 0}

    for i,day in enumerate(results):
        pred_dict = results[day]
        gt_box = pred_dict["gt_bbox_center"]
        pred_box = pred_dict["pred_bbox_center"]
        gt_length = pred_dict["gt_bbox_size"][0]
        pred_length = pred_dict["pred_bbox_size"][0]

        error_dict['x'] += (abs((abs(pred_box[0]-gt_box[0]))/gt_length)) * 100.0
        error_dict['y'] += (abs((abs(pred_box[1]-gt_box[1]))/gt_length)) * 100.0
        error_dict['z'] += (abs((abs(pred_box[2]-gt_box[2]))/gt_length)) * 100.0
        error_dict['l'] += (abs((abs(pred_length - gt_length))/gt_length)) * 100.0
        error_dict['num_imgs'] += 1

    print("-------------------------------------")
    print("X Error: ", error_dict['x'] / error_dict['num_imgs'])
    print("Y Error: ", error_dict['y'] / error_dict['num_imgs'])
    print("Z Error: ", error_dict['z'] / error_dict['num_imgs'])
    print("Lenght Error: ", error_dict['l'] / error_dict['num_imgs'])
    print("-------------------------------------\n")

def calc_errors_on_closest_bbox_human(results, results_all, human_pare_all):
    error_dict = {'x' : 0, 'y' : 0, 'z': 0, 'l': 0 , 'num_imgs' : 0}
    counter = 0
    for day in results:
        img_path = os.path.join("/data/xiwang/behave/sequences", day)
        img = cv2.imread(img_path)
        pred_dict = results[day]
        pred_all = results_all[day]
        
        gt_box = pred_dict["gt_bbox_center"]
        gt_length = pred_dict["gt_bbox_size"][0]
       
        try:
            pred_human= human_pare_all[day]
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

        error_dict['x'] += (abs((abs(pred_box[0]-gt_box[0]))/gt_length)) * 100.0
        error_dict['y'] += (abs((abs(pred_box[1]-gt_box[1]))/gt_length)) * 100.0
        error_dict['z'] += (abs((abs(pred_box[2]-gt_box[2]))/gt_length)) * 100.0
        error_dict['l'] += (abs((abs(pred_length[0] - gt_length))/gt_length)) * 100.0
        error_dict['num_imgs'] += 1

        #log_bboxes(img, pred_box, pred_length, pred_pose, pred_cat, pred_score, human_center, pred_human["pred_bbox_size"], pred_human["pred_bbox_orientation"])
        log_bboxes(img, gt_box, pred_dict["gt_bbox_size"], pred_pose, pred_cat, pred_score, human_center, pred_human["pred_bbox_size"], pred_human["pred_bbox_orientation"])
    
    print("-------------------------------------")
    print("X Error: ", error_dict['x'] / error_dict['num_imgs'])
    print("Y Error: ", error_dict['y'] / error_dict['num_imgs'])
    print("Z Error: ", error_dict['z'] / error_dict['num_imgs'])
    print("Lenght Error: ", error_dict['l'] / error_dict['num_imgs'])
    print(f"Person not detected {counter} times")
    print("-------------------------------------\n")

def calc_errors_on_closest_bbox_human_modified(results, results_all, human_pare_all):
    error_dict = {'x' : 0, 'y' : 0, 'z': 0, 'l': 0 , 'num_imgs' : 0}
    counter = 0
    for day in results:
        pred_dict = results[day]
        pred_all = results_all[day]
        
        gt_box = pred_dict["gt_bbox_center"]
        gt_length = pred_dict["gt_bbox_size"][0]
       
        try:
            pred_human= human_pare_all[day]
            if pred_dict["pred_bbox_score"] > 0.9:
                pred_box = pred_dict["pred_bbox_center"]
                pred_length = pred_dict["pred_bbox_size"][0]
            elif pred_human["pred_bbox_score"] > 0.2:
                human_center = pred_human["pred_bbox_center"]

                object_dist_list = []
                for i, bbox in enumerate(pred_all["bbox_center"]):
                    #print("human distance: ",math.dist(human_center, bbox), " Confidence: ", (1-pred_all["bbox_score"][i]))
                    object_dist_list.append(math.dist(human_center, bbox) + (1-pred_all["bbox_score"][i]))

                pos, element = min(enumerate(object_dist_list), key=itemgetter(1))
                pred_box = pred_all["bbox_center"][pos]
                pred_length = pred_all["bbox_size"][pos][0]
            elif pred_dict["pred_bbox_score"] < 0.2:
                human_center = pred_human["pred_bbox_center"]

                object_dist_list = []
                for i, bbox in enumerate(pred_all["bbox_center"]):
                    object_dist_list.append(math.dist(human_center, bbox))

                pos, element = min(enumerate(object_dist_list), key=itemgetter(1))
                pred_box = pred_all["bbox_center"][pos]
                pred_length = pred_all["bbox_size"][pos][0]
            else:
                counter+=1
                pred_box = pred_dict["pred_bbox_center"]
                pred_length = pred_dict["pred_bbox_size"][0]
        except:
            #counter+=1
            pred_box = pred_dict["pred_bbox_center"]
            pred_length = pred_dict["pred_bbox_size"][0]

        error_dict['x'] += (abs((abs(pred_box[0]-gt_box[0]))/gt_length)) * 100.0
        error_dict['y'] += (abs((abs(pred_box[1]-gt_box[1]))/gt_length)) * 100.0
        error_dict['z'] += (abs((abs(pred_box[2]-gt_box[2]))/gt_length)) * 100.0
        error_dict['l'] += (abs((abs(pred_length - gt_length))/gt_length)) * 100.0
        error_dict['num_imgs'] += 1
    
    print("-------------------------------------")
    print("X Error: ", error_dict['x'] / error_dict['num_imgs'])
    print("Y Error: ", error_dict['y'] / error_dict['num_imgs'])
    print("Z Error: ", error_dict['z'] / error_dict['num_imgs'])
    print("Lenght Error: ", error_dict['l'] / error_dict['num_imgs'])
    print(f"Person not detected {counter} times")
    print("-------------------------------------\n")

def calc_errors_on_closest_bbox_human_by_class_relative(results, results_all, human_pare_all):
    #error_dict = {'x' : 0, 'y' : 0, 'z': 0, 'l': 0 , 'num_imgs' : 0}
    error_dict = {}
    classes = set({'backpack', 'basketball', 'boxlarge', 'boxlong', 'boxmedium','boxsmall', 'boxtiny', 'chairblack','chairwood', 'keyboard', 'monitor', 'plasticcontainer', 'stool', 'suitcase', 'tablesmall', 'tablesquare', 'toolbox', 'trashbin', 'yogaball', 'yogamat'})
    for cat in classes:
        error_dict[cat] = {'x' : 0, 'y' : 0, 'z': 0, 'l': 0 , 'gt_l': 0,  'num_imgs' : 0}
    
    counter = 0
    for day in results:
        pred_dict = results[day]
        pred_all = results_all[day]
        cat_curr = (day.split("/")[0]).split("_")[2]
        
        gt_box = pred_dict["gt_bbox_center"]
        gt_length = pred_dict["gt_bbox_size"][0]

        try:
            pred_human= human_pare_all[day]
            human_center = pred_human["pred_bbox_center"]

            object_dist_list = []
            for i, bbox in enumerate(pred_all["bbox_center"]):
                #print("human distance: ",math.dist(human_center, bbox), " Confidence: ", (1-pred_all["bbox_score"][i]))
                object_dist_list.append(math.dist(human_center, bbox) + (1-pred_all["bbox_score"][i]))

            pos, element = min(enumerate(object_dist_list), key=itemgetter(1))
            pred_box = pred_all["bbox_center"][pos]
            pred_length = pred_all["bbox_size"][pos][0]
        except:
            counter+=1
            pred_box = pred_dict["pred_bbox_center"]
            pred_length = pred_dict["pred_bbox_size"][0]

        error_dict[cat_curr]['x'] += (abs((abs(pred_box[0]-gt_box[0]))/gt_length)) * 100.0
        error_dict[cat_curr]['y'] += (abs((abs(pred_box[1]-gt_box[1]))/gt_length)) * 100.0
        error_dict[cat_curr]['z'] += (abs((abs(pred_box[2]-gt_box[2]))/gt_length)) * 100.0
        error_dict[cat_curr]['l'] += (abs((abs(pred_length - gt_length))/gt_length)) * 100.0
        error_dict[cat_curr]['gt_l'] += gt_length
        error_dict[cat_curr]['num_imgs'] += 1
    
    for cat in classes:
        print("-------------------------------------")
        print("CLASS: ", cat)
        print("Lenght of the object: ", error_dict[cat]['gt_l'] / error_dict[cat]['num_imgs'])
        print("X Error: ", error_dict[cat]['x'] / error_dict[cat]['num_imgs'])
        print("Y Error: ", error_dict[cat]['y'] / error_dict[cat]['num_imgs'])
        print("Z Error: ", error_dict[cat]['z'] / error_dict[cat]['num_imgs'])
        print("Lenght Error: ", error_dict[cat]['l'] / error_dict[cat]['num_imgs'])
        #print(f"Person not detected {counter} times")
        print("-------------------------------------\n")

def calc_errors_on_closest_bbox_human_by_class_absolute(results, results_all, human_pare_all):
    #error_dict = {'x' : 0, 'y' : 0, 'z': 0, 'l': 0 , 'num_imgs' : 0}
    error_dict = {}
    classes = set({'backpack', 'basketball', 'boxlarge', 'boxlong', 'boxmedium','boxsmall', 'boxtiny', 'chairblack','chairwood', 'keyboard', 'monitor', 'plasticcontainer', 'stool', 'suitcase', 'tablesmall', 'tablesquare', 'toolbox', 'trashbin', 'yogaball', 'yogamat'})
    for cat in classes:
        error_dict[cat] = {'x' : 0, 'y' : 0, 'z': 0, 'l': 0 , 'gt_l': 0,  'num_imgs' : 0}
    
    counter = 0
    for day in results:
        pred_dict = results[day]
        pred_all = results_all[day]
        cat_curr = (day.split("/")[0]).split("_")[2]
        
        gt_box = pred_dict["gt_bbox_center"]
        gt_length = pred_dict["gt_bbox_size"][0]

        try:
            pred_human= human_pare_all[day]
            human_center = pred_human["pred_bbox_center"]

            object_dist_list = []
            for i, bbox in enumerate(pred_all["bbox_center"]):
                #print("human distance: ",math.dist(human_center, bbox), " Confidence: ", (1-pred_all["bbox_score"][i]))
                object_dist_list.append(math.dist(human_center, bbox) + (1-pred_all["bbox_score"][i]))

            pos, element = min(enumerate(object_dist_list), key=itemgetter(1))
            pred_box = pred_all["bbox_center"][pos]
            pred_length = pred_all["bbox_size"][pos][0]
        except:
            counter+=1
            pred_box = pred_dict["pred_bbox_center"]
            pred_length = pred_dict["pred_bbox_size"][0]

        error_dict[cat_curr]['x'] += (abs(pred_box[0]-gt_box[0]))
        error_dict[cat_curr]['y'] += (abs(pred_box[1]-gt_box[1]))
        error_dict[cat_curr]['z'] += (abs(pred_box[2]-gt_box[2]))
        error_dict[cat_curr]['l'] += (abs(pred_length - gt_length))
        error_dict[cat_curr]['gt_l'] += gt_length
        error_dict[cat_curr]['num_imgs'] += 1
    
    for cat in classes:
        print("-------------------------------------")
        print("CLASS: ", cat)
        print("Lenght of the object: ", error_dict[cat]['gt_l'] / error_dict[cat]['num_imgs'])
        print("X Error: ", error_dict[cat]['x'] / error_dict[cat]['num_imgs'])
        print("Y Error: ", error_dict[cat]['y'] / error_dict[cat]['num_imgs'])
        print("Z Error: ", error_dict[cat]['z'] / error_dict[cat]['num_imgs'])
        print("Lenght Error: ", error_dict[cat]['l'] / error_dict[cat]['num_imgs'])
        #print(f"Person not detected {counter} times")
        print("-------------------------------------\n")
 
def calc_chamfer_on_different_iou(data_path):
        all_images_dict = json.load(open(os.path.join(data_path,"per_img_result_2.json")))
        low_iou_images = set()
        detectable_classes  =set()
        low_iou_dict = {'chamfer_human': [], 'chamfer_object' : [], 'num_imgs': 0}
        high_iou_dict = {'chamfer_human': [], 'chamfer_object' : [], 'num_imgs': 0}

        files_path = os.path.join(data_path, "behave_iou")

        original_files = [x for x in glob("%s/*.txt" % files_path) if "_original" in x]

        for file in original_files:
            detectable_classes.add((file.split("/")[-1]).split("_")[0])
            f = open(file, 'r')
            while(True):
                image = f.readline()
                if not image:
                    break
                if "k1" in image:
                    low_iou_images.add((image.split("/")[-1])[:-13])

        
        for image in all_images_dict.keys():
            if image.split("_")[2] in detectable_classes:
                if image.replace('-', '') in low_iou_images:
                    low_iou_dict['chamfer_object'].append(all_images_dict[image][0])
                    low_iou_dict['chamfer_human'].append(all_images_dict[image][1])
                    low_iou_dict['num_imgs'] +=1
                else:
                    high_iou_dict['chamfer_object'].append(all_images_dict[image][0])
                    high_iou_dict['chamfer_human'].append(all_images_dict[image][1])
                    high_iou_dict['num_imgs'] +=1             
        

        print("-------------------------------------")
        #print("Human IOU < 0.3 mean: ", np.mean(low_iou_dict['chamfer_human']))
        #print("Human IOU < 0.3 std: ", np.std(low_iou_dict['chamfer_human']))
        print("Object IOU < 0.3 mean: ", np.mean(low_iou_dict['chamfer_object']))
        print("Object IOU < 0.3 std: ", np.std(low_iou_dict['chamfer_object']))
        #print("Human IOU > 0.3 mean: ",np.mean(high_iou_dict['chamfer_human']))
        #print("Human IOU > 0.3 std: ",np.std(high_iou_dict['chamfer_human']))
        print("Object IOU > 0.3 mean: ",np.mean(high_iou_dict['chamfer_object']))
        print("Object IOU > 0.3 std: ",np.std(high_iou_dict['chamfer_object']))
        print(f"Low IOU images: {low_iou_dict['num_imgs']}, High IOU image: {high_iou_dict['num_imgs']}, Total images: {low_iou_dict['num_imgs'] + high_iou_dict['num_imgs']}")
        print("-------------------------------------")

def calc_iou_on_3d_bbox(results, results_all, human_pare_all):
    boxes_gt, boxes_pred = [], []
    device = (
                torch.device("cuda:0") 
                if torch.cuda.is_available()
                else torch.device("cpu")
        )
    for idx,day in enumerate(results):
        pred_dict = results[day]
        pred_all = results_all[day]
        
        gt_box = pred_dict["gt_bbox_center"]
        gt_length = pred_dict["gt_bbox_size"][0]
       
        boxes_gt.append([[gt_box[0] - gt_length/2.0, gt_box[1] - gt_length/2.0, gt_box[2] - gt_length/2.0], [gt_box[0] + gt_length/2.0, gt_box[1] - gt_length/2.0, gt_box[2] - gt_length/2.0],
                         [gt_box[0] + gt_length/2.0, gt_box[1] + gt_length/2.0, gt_box[2] - gt_length/2.0], [gt_box[0] - gt_length/2.0, gt_box[1] + gt_length/2.0, gt_box[2] - gt_length/2.0],
                         [gt_box[0] - gt_length/2.0, gt_box[1] - gt_length/2.0, gt_box[2] + gt_length/2.0], [gt_box[0] + gt_length/2.0, gt_box[1] - gt_length/2.0, gt_box[2] + gt_length/2.0],
                         [gt_box[0] + gt_length/2.0, gt_box[1] + gt_length/2.0, gt_box[2] + gt_length/2.0], [gt_box[0] - gt_length/2.0, gt_box[1] + gt_length/2.0, gt_box[2] + gt_length/2.0]])
        try:
            pred_human= human_pare_all[day]
            human_center = pred_human["pred_bbox_center"]

            object_dist_list = []
            for i, bbox in enumerate(pred_all["bbox_center"]):
                #print("human distance: ",math.dist(human_center, bbox), " Confidence: ", (1-pred_all["bbox_score"][i]))
                object_dist_list.append(math.dist(human_center, bbox) + (1-pred_all["bbox_score"][i]))

            pos, element = min(enumerate(object_dist_list), key=itemgetter(1))
            pred_box = pred_all["bbox_center"][pos]
            pred_length = pred_all["bbox_size"][pos][0]
        except:
            #counter+=1
            pred_box = pred_dict["pred_bbox_center"]
            pred_length = pred_dict["pred_bbox_size"][0]
        
        boxes_pred.append([[pred_box[0] - pred_length/2.0, pred_box[1] - pred_length/2.0, pred_box[2] - pred_length/2.0], [pred_box[0] + pred_length/2.0, pred_box[1] - pred_length/2.0, pred_box[2] - pred_length/2.0],
                           [pred_box[0] + pred_length/2.0, pred_box[1] + pred_length/2.0, pred_box[2] - pred_length/2.0], [pred_box[0] - pred_length/2.0, pred_box[1] + pred_length/2.0, pred_box[2] - pred_length/2.0],
                           [pred_box[0] - pred_length/2.0, pred_box[1] - pred_length/2.0, pred_box[2] + pred_length/2.0], [pred_box[0] + pred_length/2.0, pred_box[1] - pred_length/2.0, pred_box[2] + pred_length/2.0],
                           [pred_box[0] + pred_length/2.0, pred_box[1] + pred_length/2.0, pred_box[2] + pred_length/2.0], [pred_box[0] - pred_length/2.0, pred_box[1] + pred_length/2.0, pred_box[2] + pred_length/2.0]])


    
    boxes_gt = torch.tensor(boxes_gt, device= device, dtype=torch.float32)
    boxes_pred = torch.tensor(boxes_pred, device= device, dtype=torch.float32)
    intersection_vol, iou_3d = box3d_overlap(boxes_gt, boxes_pred)

    batch_size = boxes_gt.shape[0]
    print(batch_size)
    iou_sum = 0.0
    for j in range(batch_size):
        iou_sum += iou_3d[j,j]

    print((iou_sum/batch_size) * 100.0)

if __name__ == "__main__":

    results = json.load(open("predictions/results_interaction.json"))["best_score vs gt"]
    results_all = json.load(open("predictions/results_interaction.json"))["all_predicted"]
    human_pare_all = json.load(open("predictions/results_interaction.json"))["person"]

    #results = json.load(open("predictions/results_person_large.json"))["best_score vs gt"]
    #results_all = json.load(open("predictions/results_person_large.json"))["all_predicted"]
    #human_pare_all = json.load(open("predictions/results_person_large.json"))["person"]

    wandb.init("bbox evaluation")

    calc_errors_on_high_prob_bbox(results)

    calc_errors_using_closest_bbox(results, results_all)

    calc_errors_on_closest_bbox_human(results, results_all, human_pare_all)

    calc_chamfer_on_different_iou("/data/aruzzi/Behave/")

    calc_iou_on_3d_bbox(results, results_all, human_pare_all)
    
    #calc_errors_on_closest_bbox_human_by_class_relative(results, results_all, human_pare_all)

    #calc_errors_on_closest_bbox_human_by_class_absolute(results, results_all, human_pare_all)

    calc_num_wrong_bbox(results)



