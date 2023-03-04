import json
from operator import itemgetter
import math
import torch
import wandb
import os
from glob import glob
from pytorch3d.ops import box3d_overlap

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
            pred_length = pred_all["bbox_size"][pos][0]
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
        all_images_dict = json.load(open(os.path.join(data_path,"per_img_result.json")))
        low_iou_images = set()
        detectable_classes  =set()
        low_iou_dict = {'chamfer_mean': 0.0, 'chamfer_std': 0.0, 'num_imgs': 0}
        high_iou_dict = {'chamfer_mean': 0.0, 'chamfer_std': 0.0, 'num_imgs': 0}

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
                #print(image.split("_")[2])
                if image.replace('-', '') in low_iou_images:
                    low_iou_dict['chamfer_mean'] += all_images_dict[image][0]
                    low_iou_dict['chamfer_std'] += all_images_dict[image][1]
                    low_iou_dict['num_imgs'] +=1
                else:
                    high_iou_dict['chamfer_mean'] += all_images_dict[image][0]
                    high_iou_dict['chamfer_std'] += all_images_dict[image][1]
                    high_iou_dict['num_imgs'] +=1             
        

        print("-------------------------------------")
        print("IOU < 0.3 mean: ", low_iou_dict['chamfer_mean'] / low_iou_dict['num_imgs'])
        print("IOU < 0.3 std: ", low_iou_dict['chamfer_std'] / low_iou_dict['num_imgs'])
        print("IOU > 0.3 mean: ", high_iou_dict['chamfer_mean'] / high_iou_dict['num_imgs'])
        print("IOU > 0.3 std: ", high_iou_dict['chamfer_std'] / high_iou_dict['num_imgs'])
        print(f"Low IOU images: {low_iou_dict['num_imgs']}, High IOU image: {high_iou_dict['num_imgs']}, Total images: {low_iou_dict['num_imgs'] + high_iou_dict['num_imgs']}")
        print("-------------------------------------")

def calc_iou_on_3d_bbox(results, results_all, human_pare_all):
    boxes_gt, boxes_pred = [], []
    device = (
                torch.device("cuda:0") 
                if torch.cuda.is_available()
                else torch.device("cpu")
        )
    for i,day in enumerate(results):
        pred_dict = results[day]
        pred_all = results_all[day]
        
        gt_box = pred_dict["gt_bbox_center"]
        gt_length = pred_dict["gt_bbox_size"][0]

        boxes_gt.append([[gt_box[0] - gt_length/2.0, gt_box[1] + gt_length/2.0, gt_box[2] - gt_length/2.0], [gt_box[0] + gt_length/2.0, gt_box[1] + gt_length/2.0, gt_box[2] - gt_length/2.0,
                         [gt_box[0] + gt_length/2.0, gt_box[1] - gt_length/2.0, gt_box[2] - gt_length/2.0]], [gt_box[0] - gt_length/2.0, gt_box[1] - gt_length/2.0, gt_box[2] - gt_length/2.0],
                         [gt_box[0] - gt_length/2.0, gt_box[1] + gt_length/2.0, gt_box[2] + gt_length/2.0], [gt_box[0] + gt_length/2.0, gt_box[1] + gt_length/2.0, gt_box[2] + gt_length/2.0],
                         [gt_box[0] + gt_length/2.0, gt_box[1] - gt_length/2.0, gt_box[2] + gt_length/2.0], [gt_box[0] - gt_length/2.0, gt_box[1] - gt_length/2.0, gt_box[2] + gt_length/2.0]])
       
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

        boxes_pred.append([[pred_box[0] - pred_length/2.0, pred_box[1] + pred_length/2.0, pred_box[2] - pred_length/2.0], [pred_box[0] + pred_length/2.0, pred_box[1] + pred_length/2.0, pred_box[2] - pred_length/2.0,
                    [pred_box[0] + pred_length/2.0, pred_box[1] - pred_length/2.0, pred_box[2] - pred_length/2.0]], [pred_box[0] - pred_length/2.0, pred_box[1] - pred_length/2.0, pred_box[2] - pred_length/2.0],
                    [pred_box[0] - pred_length/2.0, pred_box[1] + pred_length/2.0, pred_box[2] + pred_length/2.0], [pred_box[0] + pred_length/2.0, pred_box[1] + pred_length/2.0, pred_box[2] + pred_length/2.0],
                    [pred_box[0] + pred_length/2.0, pred_box[1] - pred_length/2.0, pred_box[2] + pred_length/2.0], [pred_box[0] - pred_length/2.0, pred_box[1] - pred_length/2.0, pred_box[2] + pred_length/2.0]])
        if i ==2:
            break
        #d = pred_box.append(pred_length)
        #g = gt_box.append(gt_length)
        #dd = torch.tensor(d, device=device, dtype=torch.float32)
        #gg = torch.tensor(g, device=device, dtype=torch.float32)
        #ious = _C.iou_box3d(dd, gg)[1].cpu().numpy()
    boxes_gt = torch.tensor(boxes_gt, device= device, dtype=torch.float32)
    boxes_pred = torch.tensor(boxes_pred, device= device, dtype=torch.float32)
    print(boxes_gt.shape, boxes_pred.shape)
    intersection_vol, iou_3d = box3d_overlap(boxes_gt, boxes_pred)
    print(iou_3d)

if __name__ == "__main__":
    #results = json.load(open("predictions/results_2.json"))["best_score vs gt"]
    #results_all = json.load(open("predictions/results_2.json"))["all_predicted"]
    results = json.load(open("predictions/results_interaction.json"))["best_score vs gt"]
    results_all = json.load(open("predictions/results_interaction.json"))["all_predicted"]
    human_pare_all = json.load(open("predictions/results_interaction.json"))["person"]

    wandb.init("bbox evaluation")

    calc_errors_on_high_prob_bbox(results)

    calc_errors_using_closest_bbox(results, results_all)

    calc_errors_on_closest_bbox_human(results, results_all, human_pare_all)

    calc_chamfer_on_different_iou("/data/aruzzi/Behave/")

    calc_iou_on_3d_bbox(results, results_all, human_pare_all)
    
    #calc_errors_on_closest_bbox_human_by_class_relative(results, results_all, human_pare_all)

    #calc_errors_on_closest_bbox_human_by_class_absolute(results, results_all, human_pare_all)

    calc_num_wrong_bbox(results)



