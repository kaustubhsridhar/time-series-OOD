import os
import numpy as np
import time
import json

from memories.memorization import memory, memorization
from crash_prediction.predict_carla import check_carla_ood, check_carla_heavy_rain_ood

import logging

np.random.seed(101)

def build_memories_lidar(source_dir, dest_dir, init_distance):

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    memorization_object = memorization(source_dir, dest_dir)
    memorization_object.learn_memories_with_CLARANS(init_distance_threshold = init_distance)

def build_memories_carla(source_dir, dest_dir, init_distance, source_list=None):

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    memorization_object = memorization(source_dir, dest_dir, source_list)
    memorization_object.learn_memories_with_CLARANS(init_distance_threshold = init_distance)


def run_carla_prediction(memory_dir, source_dir, initial_memory_threshold, detect_threshold,prob_threshold,window_size,window_threshold,task):

    memorization_object = memorization(None, memory_dir)
    memorization_object.load_memories(expand_radius = 0.05)
    if task == "heavy_rain":
        print("**************************************************************")
        f = open("./results/carla_"+task+"_exp_results.txt", "a")
        print("(W: %s tau: %s alpha: %s dist: %s) " % (str(window_size),str(window_threshold),str(prob_threshold),str(initial_memory_threshold)))
        f.write("(W: {} tau: {} alpha: {} dist: {} ) ".format(str(window_size),str(window_threshold),str(prob_threshold),str(initial_memory_threshold)))
        print("Mem: ",len(memorization_object.current_memory_dictionary))
        f.write("Mem: {} ".format(str(len(memorization_object.current_memory_dictionary))))
        #run in distribution experiment
        in_source_dir = source_dir + os.sep + "in_test"
        stats = check_carla_heavy_rain_ood(in_source_dir, memorization_object, initial_memory_threshold, window_size,int(window_threshold),detect_threshold,prob_threshold)  
        #write to file
        with open("./results/ood_result"+"_in_"+task+"_"+str(initial_memory_threshold)+"_"+str(window_size)+"_"+str(prob_threshold)+"_"+str(window_threshold)+".json", 'w') as outfile:
                json.dump(stats, outfile)
        outfile.close()
        
        print("FP: %d/%d" %(stats["ood_episode"],stats["total_episode"]))
        f.write("FP: {}/{}" .format(str(stats["ood_episode"]),str(stats["total_episode"])))
        #run out of distribution experiment
        out_source_dir = source_dir + os.sep + "oods_heavy_rain"
        stats = check_carla_heavy_rain_ood(out_source_dir, memorization_object, initial_memory_threshold, window_size,int(window_threshold),detect_threshold,prob_threshold)  
        #write to file
        if(len(stats["detect_frame_list"])>0):
            print("FN: %d/%d Avg Delay: %f Exec Time: %f "%(stats["total_episode"]-stats["ood_episode"],stats["total_episode"],stats["average_window_delay"],stats["average_evaluate_time"]))
            f.write("FN: {}/{} Avg Delay: {} Exec Time: {} \n".format(str(stats["total_episode"]-stats["ood_episode"]),str(stats["total_episode"]),str(stats["average_window_delay"]),str(stats["average_evaluate_time"]) ))
        else:
            print("FN: %d/%d Avg Delay: N/A Exec Time: N/A "%(stats["total_episode"]-stats["ood_episode"],stats["total_episode"]))
            f.write("FN: {}/{} Avg Delay: N/A \n Exec Time: N/A".format(str(stats["total_episode"]-stats["ood_episode"]),str(stats["total_episode"]) ))
            stats["average_window_delay"]=None
        f.close()
        with open("./results/ood_result"+"_out_"+task+"_"+str(initial_memory_threshold)+"_"+str(window_size)+"_"+str(prob_threshold)+"_"+str(window_threshold)+".json", 'w') as outfile:
                json.dump(stats, outfile)
        outfile.close()
    else:
        stats = check_carla_ood(source_dir, memorization_object, initial_memory_threshold, window_size,int(window_threshold),detect_threshold,prob_threshold, task)
        #write to file
        print("**************************************************************")
        f = open("./results/carla_"+task+"_exp_results.txt", "a")
        print("(W: %s tau: %s alpha: %s dist: %s) " % (str(window_size),str(window_threshold),str(prob_threshold),str(initial_memory_threshold)))
        f.write("(W: {} tau: {} alpha: {} dist: {} ) ".format(str(window_size),str(window_threshold),str(prob_threshold),str(initial_memory_threshold)))
        if(len(stats["detect_frame_list"])>0):
            print("FN: %d/%d Avg Delay: %f "%(stats["total_episode"]-stats["ood_episode"],stats["total_episode"],stats["average_window_delay"]))
            f.write("FN: {}/{} Avg Delay: {} \n".format(str(stats["total_episode"]-stats["ood_episode"]),str(stats["total_episode"]),str(stats["average_window_delay"]) ))
        else:
            print("FN: %d/%d Avg Delay: N/A "%(stats["total_episode"]-stats["ood_episode"],stats["total_episode"]))
            f.write("FN: {}/{} Avg Delay: N/A \n".format(str(stats["total_episode"]-stats["ood_episode"]),str(stats["total_episode"]) ))
        f.close()
        with open("./results/ood_result"+"_"+task+"_"+str(initial_memory_threshold)+"_"+str(window_size)+"_"+str(prob_threshold)+"_"+str(window_threshold)+".json", 'w') as outfile:
            json.dump(stats, outfile)
        outfile.close()
            
    return stats

def dump_distances(memory_dir):
    memorization_object = memorization(None, memory_dir)
    memorization_object.load_memories(expand_radius = 0.05)
    memorization_object.dump_memory_distance(memory_dir)