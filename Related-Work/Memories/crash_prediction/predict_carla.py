import os
import random
import csv
import time
from tqdm import tqdm
import json

random.seed(201)

def myFunc(e):
    return int((e.split("/")[-1]).split(".")[0])

def read_image(folder,img_folder):
    img_list = []
    for root, dirs, files in os.walk(img_folder):
        for file_ in files:
            if(file_.endswith(".csv")):
                with open(os.path.join(root,file_), newline='') as csvfile:
                    csv_reader = csv.reader(csvfile, delimiter=',')
                    for row in csv_reader:
                        temp = row[0].split("/")
                        if("(1)" not in temp[-1] and os.path.exists(os.path.join(root, temp[-1]+".png")) ):
                            img_list.append(os.path.join(root, temp[-1]+".png"))
    img_list.sort(key=myFunc)
    return img_list

def read_precipitation(folder,exp_num, end_threshold):
    end_point = 0
    precipitation_dict={}
    with open("./"+folder+"/"+str(exp_num)+'/label.csv', newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            temp = row[0].split("/")
            mod_path = str(exp_num)+"/"+temp[-1]+".png"
            precipitation_dict[mod_path] = float(row[-1])
            if(float(row[-1]) <= end_threshold):
                end_point=int(row[0].split("/")[-1])
    csvfile.close()
    return precipitation_dict,end_point

def read_image_for_carla(img_folder):
    exp_list = []
    for root, dirs, files in os.walk(img_folder):
        for i in dirs:
            if i not in exp_list:
                exp_list.append(i)
    return exp_list

def check_carla_ood(exp_folder,memorization_object,initial_memory_threshold, window_size,window_thres,detect_threshold,prob_threshold,task):
    exp_list = read_image_for_carla(exp_folder)
    total=0
    total_detect = 0
    ood_epi = 0
    total_delay = []
    detect_res_list =[]
    detect_frame_list =[]
    threshold_list={"oods_bike":20,"oods_foggy":0,"oods_night":10,"out_foggy":0,"out_night":0,"out_rainy":0,"out_snowy":0,"out_replay":8,"in":200}

    ood_window = 0
    num_window = 0
    gt_ood_window = 0

    result=[]
    gt_result = []


    for exp_num in tqdm(exp_list):
        window = []
        total_exp_time = []
        episode = False
        window_delay = 0
        #mapping img to the memory
        img_list = read_image(exp_folder,"./"+exp_folder+"/"+str(exp_num))

        for img_path in img_list:
            

            current_frame = int(img_path.split("/")[-1][:-4])
            total += 1
            key = img_path.split("/")[-2]+"/"+img_path.split("/")[-1]
            start_ = time.time()
            nearest_memory, matched_set, prob_density, exp_time_ = memorization_object.find_match(img_path,initial_memory_threshold)

            

            if (len(window) == window_size):
                window.pop(0)
                if (prob_density < prob_threshold):
                    window.append(0)
                else:
                    window.append(1)
                
                    
            else:
                if (prob_density < prob_threshold):
                    window.append(0)
                else:
                    window.append(1)

            if (len(window) == window_size):
                if task == 'in':
                    gt_result.append(1)
                else:
                    gt_result.append(0)
                num_window += 1
                total_win = window.count(0)
                if (total_win >= window_thres) and (episode == False):
                    episode = True
                    total_detect += 1
                    detect_frame_list.append(current_frame-15)
                if (total_win >= window_thres):
                    ood_window += 1
                    result.append(0)
                else:
                    result.append(1)

            #test ood episode for only
            total_exp_time.append(round((time.time()-start_)*1000,2)) 
            #if(episode):
            #    break
            
        detect_res_list.append(episode)
        total_delay.append(window_delay)
        #print(exp_num,episode)
        if (episode):
            ood_epi += 1
    
    with open('test_'+task+'.json', 'w') as f:
        json.dump({'gt':gt_result, 'pred':result}, f)
    f.close()

    print(ood_window, num_window, ood_window/num_window)
    results_stat = {}
    
    results_stat["detection_rate"] = round(ood_epi/len(exp_list),3)
    results_stat["ood_episode"] = ood_epi
    results_stat["total_episode"] = len(exp_list)
    results_stat["detect_frame_list"] = detect_frame_list
    results_stat["detect_res_list"] = detect_res_list
    new_frame=[]
    for m in results_stat["detect_frame_list"]:
        if m > threshold_list[task]:
            new_frame.append(m-threshold_list[task])
        else:
            new_frame.append(0)
    results_stat["window_list"] = new_frame
    if (len(results_stat["detect_frame_list"]) > 0):
        results_stat["average_window_delay"] = round(sum(results_stat["window_list"])/len(results_stat["detect_frame_list"]),2)
    else:
        results_stat["average_window_delay"] = None

    return results_stat

def check_carla_heavy_rain_ood(exp_folder,memorization_object,initial_memory_threshold, window_size,window_thres,detect_threshold,prob_threshold):
    exp_list = read_image_for_carla(exp_folder)
    frame_diff = 0
    total_prep = 0
    total=0
    total_detect = 0
    ood_epi = 0
    total_delay = []
    evaluate_time_list =[]
    total_evaluate_time_list =[]
    detect_pre_list =[]
    detect_frame_list =[]

    ood_window = 0
    num_window = 0
    gt_ood_window = 0

    for exp_num in tqdm(exp_list):
        window = []
        exp_time = []   
        total_exp_time = []
        episode = False
        window_delay = 0

        #mapping img to the memory
        
        precipitation_dict,end_point=read_precipitation(exp_folder,exp_num,detect_threshold)
        img_list = read_image(exp_folder,"./"+exp_folder+"/"+str(exp_num))

        for img_path in img_list:
            num_window += 1
            print(img_path)
            current_frame = int(img_path.split("/")[-1][:-4])
            total += 1
            key = img_path.split("/")[-2]+"/"+img_path.split("/")[-1]
            start_ = time.time()
            nearest_memory, matched_set, prob_density, exp_time_ = memorization_object.find_match(img_path,initial_memory_threshold)
            
            exp_time.append(round((exp_time_)*1000,5))
            if (len(window) >= window_size):
                window.pop(0)
                if (prob_density < prob_threshold):
                    window.append(0)
                else:
                    window.append(1)
                total_win = window.count(0)
                if (total_win >= window_thres):
                    ood_window += 1
                #if (total_win >= window_thres and episode == False):
                #    episode = True
                #    total_detect += 1
                #    detect_pre = int(precipitation_dict[key])
                #    detect_pre_list.append(detect_pre)
                #    detect_frame_list.append(current_frame)
                #    total_prep += detect_pre
                    
            else:
                if (prob_density < prob_threshold):
                    window.append(0)
                else:
                    window.append(1)

            #test ood episode for only
            total_exp_time.append(round((time.time()-start_)*1000,2)) 
            if (current_frame - end_point > 0):
                gt_ood_window += 1
            if(current_frame - end_point > 0 and episode == False):
                window_delay += 1

            if(current_frame - end_point > 1 and episode == True):
                frame_diff += window_delay
                #break
        evaluate_time_list.append(exp_time)
        total_evaluate_time_list.append(total_exp_time)
        total_delay.append(window_delay)
        if (episode):
            ood_epi += 1

    print(ood_window, num_window, ood_window/num_window)

    results_stat = {}

    results_stat["detection_rate"] = 100*round(ood_epi/len(exp_list),3)
    results_stat["ood_episode"] = ood_epi
    results_stat["total_episode"] = len(exp_list)
    results_stat["detect_frame_list"] = detect_frame_list
    
    if total_detect > 0:
        results_stat["average_window_delay"] = round(frame_diff/total_detect,2)
    else: 
        results_stat["average_window_delay"] = 0


    total_time = 0
    length = 0
    #select 3 random traces
    select = random.sample(range(0, len(evaluate_time_list)), 3)
    for m in select:
        total_time += sum(evaluate_time_list[m]) 
        length += len(evaluate_time_list[m])
    results_stat["average_evaluate_time"] = round(total_time/length,2)

    

    return results_stat

