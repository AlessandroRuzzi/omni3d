import json
from operator import itemgetter
import math
import torch

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
                print(math.dist(human_center, bbox))
                object_dist_list.append(math.dist(human_center, bbox) + (1-pred_all["bbox_score"][i]))

            pos, element = min(enumerate(object_dist_list), key=itemgetter(1))
            pred_box = pred_all["bbox_center"][pos]
            pred_length = pred_all["bbox_size"][pos][0]
        except:
            counter+=1
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
 

if __name__ == "__main__":
    results = json.load(open("predictions/results_person_large.json"))["best_score vs gt"]
    results_all = json.load(open("predictions/results_person_large.json"))["all_predicted"]
    human_pare_all = json.load(open("predictions/results_person_large.json"))["person"]
    

    calc_errors_on_high_prob_bbox(results)

    calc_errors_using_closest_bbox(results, results_all)

    calc_errors_on_closest_bbox_human(results, results_all, human_pare_all)

    calc_num_wrong_bbox(results)



