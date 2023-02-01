#
# Code written by : Souradeep Dutta,
#  duttaso@seas.upenn.edu, souradeep.dutta@colorado.edu
# Website : https://sites.google.com/site/duttasouradeep39/
#

import os
import json
import numpy as np
from shutil import copyfile
import copy
import random
import torch
import time
import csv

from torchvision import transforms, datasets
from copy import deepcopy as dc
from collections import OrderedDict

from distance_calculations.find_features import return_feature_vector
from distance_calculations.pytorch_modified_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from memories.memory import memory
from memories.data import data
from memories.carla_data import carla_data

f_zeros = 4
random.seed(1)

verbosity = False

def epanechnikov_kernel(x,bandwidth):
    kernel = 'epanechnikov'
    if((x/bandwidth) < 1):
        y = (3/(4*bandwidth))*(1-(x**2)/(bandwidth**2))
    else:
        y = 0
    return y


class memorization :


    def __init__(self, source_dir = None, saving_dir = None, source_list=None):

        assert saving_dir

        self.source_dir = source_dir
        self.memory_dir = saving_dir

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory_suffix = "_memory"


        t0 = time.time()

        if self.source_dir:

            self.data_container = {}

            for root, dirs, files in os.walk(self.source_dir):
                print("root",root)

                if source_list:
                    if root not in source_list:
                        print("not in list",root)
                        continue
                
                if("in_test" in root):
                    print(root)
                    with open(str(root)+'/label.csv', newline='') as csvfile:
                        csv_reader = csv.reader(csvfile, delimiter=',')
                        for row in csv_reader:
                            temp = row[0].split("/")
                            mod_path = str(root)+"/"+temp[-1]+".png"
                            filename = os.path.join(root,temp[-1]+".png")
                            current_data = carla_data(self.device)
                            current_data.create_data_from_scan(filename)
                            self.data_container[filename] = \
                            {"data": current_data, "files": {"image" : filename} }
                                
                    continue
                for file in files:
                    if(file.endswith("png") ):

                        filename = os.path.join(root,file)

                        current_data = carla_data(self.device)
                        current_data.create_data_from_scan(filename)
                        self.data_container[filename] = \
                        {"data": current_data, "files": {"image" : filename} }
                    

            t1 = time.time()

        self.current_memory_dictionary = {}

    def learn_memories(self, distance_threshold, save_to_disk = True):

        assert self.source_dir

        if verbosity:
            print("Learning initial memories - ")

        unsolved_set = set()
        for name in self.data_container.keys() :
            unsolved_set.add(name)

        self.current_memory_dictionary = {}

        memory_index = 1
        while len(unsolved_set) > 0 :
            dir_name = str(memory_index).zfill(f_zeros) + self.memory_suffix
            current_memory_dir = os.path.join(self.memory_dir, dir_name)


            random_memory_candidate = random.choice(list(unsolved_set))
            unsolved_set.remove(random_memory_candidate)

            current_memory = memory(self.device)
            current_memory.data_point = dc(self.data_container[random_memory_candidate]["data"])
            current_memory.data_point_filename = dc(self.data_container[random_memory_candidate]["files"]["image"])
            current_memory.samples_list.append(self.data_container[random_memory_candidate]["files"])
            current_memory.distance_score = [distance_threshold]

            unsolved_set, _ = current_memory.apply_memory(self.data_container, unsolved_set)

            self.current_memory_dictionary[current_memory_dir] = current_memory

            memory_index += 1


        if save_to_disk :
            for current_memory_dir in self.current_memory_dictionary.keys():
                if not os.path.exists(current_memory_dir):
                    os.mkdir(current_memory_dir)
                self.current_memory_dictionary[current_memory_dir].save_memory(current_memory_dir)




    def learn_memories_with_CLARANS(self, init_distance_threshold = 0.3, local_steps = 10, global_steps = 5, save_to_disk = True):

        best_memory_dictionary = {}
        best_cost = np.inf


        for global_step in range(global_steps):

            self.current_memory_dictionary.clear()

            # First initialize with the structure with crude choices about the global memory choice
            self.learn_memories(init_distance_threshold, save_to_disk = False)
            assert any(self.current_memory_dictionary)

            memories_count = len(self.current_memory_dictionary)

            if verbosity :
                print("Number of memories learnt - ", memories_count)

            # This distance maps every data file name to a list of medoid names, in ascending order of distance
            medoid_to_data_distance, data_to_medoid_distance = self.compute_all_distances()
            current_cost = self.compute_partitioning_cost(data_to_medoid_distance)

            local_medoid_to_data_distance = dc(medoid_to_data_distance)
            local_data_to_medoid_distance = dc(data_to_medoid_distance)

            new_cost, new_memory_dictionary = self.do_local_search(current_cost, local_medoid_to_data_distance,
                                                                   local_data_to_medoid_distance, local_steps)

            # If new cost is less than current best cost, take the change
            if (new_cost is not None) and new_cost < current_cost :
                current_cost = new_cost

            if current_cost < best_cost:
                best_cost = current_cost
                if new_memory_dictionary is not None:
                    best_memory_dictionary.clear()
                    best_memory_dictionary = dc(new_memory_dictionary)
                else:
                    best_memory_dictionary.clear()
                    best_memory_dictionary = dc(self.current_memory_dictionary)



        if any(best_memory_dictionary):
            self.current_memory_dictionary.clear()
            self.current_memory_dictionary = dc(best_memory_dictionary)


        # This find the distance of furthest data point, and then adds delta to that distance metric.

        for current_memory_dir in self.current_memory_dictionary.keys():
            current_memory = self.current_memory_dictionary[current_memory_dir]
            if save_to_disk :
                if not os.path.exists(current_memory_dir):
                    os.mkdir(current_memory_dir)
                current_memory.save_memory(current_memory_dir)
        print("Total number of memories: ", len(self.current_memory_dictionary))


    def do_local_search(self, initial_cost, medoid_to_data_dist, data_to_medoid_dist, local_steps):

        current_cost = initial_cost

        for local_step in range(local_steps):


            # Choose a random medoid
            random_medoid = random.choice(list(medoid_to_data_dist.keys()))

            # Choose a random data point in that medoid's cluster,
            # and exchange the medoid with some random data point in the cluster
            alternate_medoid = self.pick_alternate_point_for_medoid(random_medoid, medoid_to_data_dist)


            if alternate_medoid == random_medoid :
                continue

            # Reassign clusters and compute the change in cost, if negative take the change
            new_medoid_to_data_dist, new_data_to_medoid_dist = self.reassign_medoids(
            random_medoid, alternate_medoid, \
            dc(medoid_to_data_dist), dc(data_to_medoid_dist))

            new_cost = self.compute_partitioning_cost(new_data_to_medoid_dist)

            if new_cost < current_cost :
                medoid_to_data_dist = new_medoid_to_data_dist
                data_to_medoid_dist = new_data_to_medoid_dist
                current_cost = new_cost
                new_memory_dictionary = self.compute_new_memory_dicitionary(\
                medoid_to_data_dist, data_to_medoid_dist)


        if current_cost < initial_cost :
            return current_cost, new_memory_dictionary


        return None, None


    def pick_alternate_point_for_medoid(self, current_medoid, medoid_to_data_dist):
        alternate_medoid = None
        assert current_medoid in medoid_to_data_dist.keys()
        alternate_medoid = random.choice(list(medoid_to_data_dist[current_medoid].keys()))
        return alternate_medoid


    def compute_new_memory_dicitionary(self, medoid_to_data_distance, data_to_medoid_distance):

        assert (len(data_to_medoid_distance) + len(medoid_to_data_distance)) == len(self.data_container)

        new_memory_dictionary = {}


        medoid_to_sample_list = {}
        medoid_to_sample_dist = {}

        for medoid in medoid_to_data_distance.keys():
            medoid_to_sample_list[medoid] = []
            medoid_to_sample_dist[medoid] = []


        # Create the medoid to data list
        for data in data_to_medoid_distance.keys() :

            if len(data_to_medoid_distance[data].keys()) != 0 :
                closest_medoid = list(data_to_medoid_distance[data].keys())[0] # Taking the closest medoid
                medoid_to_sample_list[closest_medoid].append(self.data_container[data]["files"])
                medoid_to_sample_dist[closest_medoid].append(medoid_to_data_distance[closest_medoid][data])

        assert len(medoid_to_sample_list) == len(medoid_to_data_distance) # Making sure you grabbed everything

        total_samples = 0
        for medoid in medoid_to_sample_dist.keys():
            total_samples += len(medoid_to_sample_dist[medoid])

        assert (total_samples + len(medoid_to_sample_dist)) == len(self.data_container)
        assert len(medoid_to_sample_list) == len(medoid_to_data_distance)

        total_samples = 0
        memory_index = 1
        for medoid in medoid_to_data_distance.keys():


            # Prepping the name
            dir_name = str(memory_index).zfill(f_zeros) + self.memory_suffix
            current_memory_dir = os.path.join(self.memory_dir, dir_name)


            # Prepping the current memory
            current_memory = memory(self.device)
            current_memory.data_point = dc(self.data_container[medoid]["data"])
            current_memory.data_point_filename = dc(self.data_container[medoid]["files"]["image"])
            current_memory.samples_list = medoid_to_sample_list[medoid]
            current_memory.samples_list.append(self.data_container[medoid]["files"])

            if any(medoid_to_sample_dist[medoid]):
                current_memory.distance_score = [float(max(medoid_to_sample_dist[medoid]))]
            else:
                current_memory.distance_score = [0.0]

            new_memory_dictionary[current_memory_dir] = current_memory

            total_samples += len(current_memory.samples_list)

            memory_index += 1


        assert total_samples == len(self.data_container)
        return new_memory_dictionary


    def reassign_medoids(self, current_medoid, new_medoid, medoid_to_data_distance, data_to_medoid_distance):
        # Apply the new medoid to all the points in the data and get the distance computed
        # Note this includes other memories as well.
        new_medoid_data = self.data_container[new_medoid]["data"]
        new_medoid_distance = new_medoid_data.compute_distance_batched(self.data_container)
        del new_medoid_distance[new_medoid]

        # Distance of medoid to other medoid being deleted, except the current medoid
        # since that is going to stick around as a data point
        for medoid in medoid_to_data_distance.keys():
            if medoid != current_medoid:
                del new_medoid_distance[medoid]

            del medoid_to_data_distance[medoid][new_medoid]

        # Replace the row for current medoid with the new medoid distances
        sorted_distances = dict(sorted(new_medoid_distance.items(), key=lambda item:item[1]))
        medoid_to_data_distance[new_medoid] = sorted_distances

        del medoid_to_data_distance[current_medoid]

        # Computing current medoid to other medoid distance
        temp_data_container = {}
        for medoid in medoid_to_data_distance.keys():
            temp_data_container[medoid] = dc(self.data_container[medoid])

        current_medoid_data = self.data_container[current_medoid]["data"]
        old_medoid_to_other_medoids_distance = current_medoid_data.compute_distance_batched(temp_data_container)

        # Adding this new point to all other medoids as well
        for some_medoid in old_medoid_to_other_medoids_distance.keys():
            if some_medoid != current_medoid :
                medoid_to_data_distance[some_medoid][current_medoid] = \
                old_medoid_to_other_medoids_distance[some_medoid]

        # Deleting this, since it's no longer a data point any more
        del data_to_medoid_distance[new_medoid]

        # For each data point, replace the current medoid distance with the new medoid distance
        for data in data_to_medoid_distance.keys():
            assert data in new_medoid_distance.keys()
            del data_to_medoid_distance[data][current_medoid]
            data_to_medoid_distance[data][new_medoid] = new_medoid_distance[data]


        # Since the older medoid is a real data point now, adding it.
        data_to_medoid_distance[current_medoid] = old_medoid_to_other_medoids_distance

        for data in data_to_medoid_distance.keys():
            sorted_distances = dict(sorted(data_to_medoid_distance[data].items(), key=lambda item:item[1]))
            data_to_medoid_distance[data] = sorted_distances

        # Doing some final sanity checking

        for medoid in medoid_to_data_distance.keys():
            for data in data_to_medoid_distance.keys():
                assert medoid in data_to_medoid_distance[data].keys()
                assert data in medoid_to_data_distance[medoid].keys()
                assert len(medoid_to_data_distance) == len(data_to_medoid_distance[data])
                assert len(medoid_to_data_distance[medoid]) == len(data_to_medoid_distance)

        return medoid_to_data_distance, data_to_medoid_distance

    def compute_all_distances(self):

        medoid_to_data_distance = OrderedDict()
        data_to_medoid_distance = OrderedDict()

        # Content of above : memory filename --> list of data point filenames

        for memory_name in self.current_memory_dictionary.keys():

            current_memory_filemame = self.current_memory_dictionary[memory_name].data_point_filename

            # Apply the memory to all the points in the data and get the distance computed
            all_distances = self.current_memory_dictionary[memory_name].data_point.compute_distance_batched(self.data_container)

            # Distance of memory to other memory being deleted
            for each_mem in self.current_memory_dictionary.keys():
                current_memory = self.current_memory_dictionary[each_mem]
                del all_distances[current_memory.data_point_filename]

            sorted_distances = dict(sorted(all_distances.items(), key=lambda item:item[1]))

            medoid_to_data_distance[current_memory_filemame] = sorted_distances

            for each_data_file in all_distances.keys():
                if each_data_file not in data_to_medoid_distance.keys():
                    data_to_medoid_distance[each_data_file] = {}

                if current_memory_filemame not in data_to_medoid_distance[each_data_file].keys():
                    data_to_medoid_distance[each_data_file][current_memory_filemame] = all_distances[each_data_file]
                else:
                    assert False



        # For each data point sort out the distance to the individual medoids
        for each_data_file in data_to_medoid_distance.keys():
            data_to_medoid_distance[each_data_file] = dict(sorted( data_to_medoid_distance[each_data_file].items() \
            , key=lambda item:item[1]))



        return medoid_to_data_distance, data_to_medoid_distance

    def compute_partitioning_cost(self, data_to_medoid_distance):

        # For each medoid has the data points pre-sorted in order of distance
        # hence computing the closest distance is essentially taking the first point
        # in the list.

        sum = 0.0
        for each_data_file in data_to_medoid_distance.keys():
            first_key = list(data_to_medoid_distance[each_data_file].keys())[0] # Taking the closest medoid
            sum += data_to_medoid_distance[each_data_file][first_key]

        return sum

    def adjust_distance_threshold(self, delta):
        for memory_name in self.current_memory_dictionary.keys():
            self.current_memory_dictionary[memory_name].distance_score[0] += float(delta)
            
        
    def load_memories(self, expand_radius = 0.52):


        assert any(self.memory_dir)

        for root, dirs, files in os.walk(self.memory_dir):
            if any(dirs):
                for each_mem in dirs:
                    if each_mem.endswith(self.memory_suffix):
                        memory_dir = os.path.join(root, each_mem)
                        current_memory = memory(self.device)
                        current_memory.read_memory(memory_dir)
                        self.current_memory_dictionary[each_mem] = current_memory

        assert any(self.current_memory_dictionary)
        self.adjust_distance_threshold(delta = expand_radius)

    def form_feature_and_image_collection(self):

        image_collection = []
        feature_collection = []
        list_index_to_memory_name = {}
        index = 0

        for each_memory in self.current_memory_dictionary.keys():

            image_collection.append(self.current_memory_dictionary[each_memory].memory_image)
            feature_collection.append(self.current_memory_dictionary[each_memory].seg_feature)
            list_index_to_memory_name[index] = each_memory

            index += 1

        image_collection = torch.stack(image_collection, dim = 0)
        return image_collection, feature_collection, list_index_to_memory_name
    
    def dump_memory_distance(self,memory_dir):
        data_dictionary = {}
        for each_memory in self.current_memory_dictionary.keys():
            #print(each_memory)
            data_dictionary[each_memory] = {"data" : self.current_memory_dictionary[each_memory].data_point,\
            "files" : {"image" : None } }
        for each_memory in self.current_memory_dictionary.keys():
            all_distances = self.current_memory_dictionary[each_memory].data_point.compute_distance_batched(data_dictionary)
            for i in all_distances.keys():
                all_distances[i]=float(all_distances[i])
            self.current_memory_dictionary[each_memory].save_distance_other_memories(os.path.join(memory_dir,each_memory),all_distances)
            


    def find_match(self, data_files,initial_memory_threshold):

        
        # Form like a dummy memory from the lesion files
        
        test_memory = memory(self.device)
    
        test_memory.create_memory_from_files(data_files)

        start = time.time()
        # Form a group of data out of all the current memories
        data_dictionary = {}
        
        for each_memory in self.current_memory_dictionary.keys():
            data_dictionary[each_memory] = {"data" : self.current_memory_dictionary[each_memory].data_point,\
            "files" : {"image" : None } }

        data_dictionary_copy = dc(data_dictionary)

        if (len(data_dictionary_copy) == 0):
            return None, {}

        all_distances = test_memory.data_point.compute_distance_batched(data_dictionary_copy)
        assert len(data_dictionary_copy) == len(all_distances)
        #  Find the distances
        # Find the closest memory

        distances = {}
        min_dist = np.inf

        closest_memory = None
        for memory_name in all_distances.keys():

            current_threshold = self.current_memory_dictionary[memory_name].distance_score[0]
            if  all_distances[memory_name] < min_dist:

                distances[memory_name] = all_distances[memory_name]
                min_dist = distances[memory_name]
                closest_memory = memory_name

            
        
        prob_density = self.probability_density_estimation(all_distances,initial_memory_threshold)

        exp_time = time.time() - start
        return closest_memory, min_dist, prob_density, exp_time
    
    def probability_density_estimation(self,all_distances,initial_memory_threshold):
        prob_density = 0.0
        total = 0
        bandwidth = initial_memory_threshold + 0.4
        
        selected_distance={}
        for k, v in sorted(all_distances.items(), key=lambda item: item[1]):
            selected_distance[k]=v
            if(len(selected_distance) >= 5):
                break

        for i in selected_distance.keys():
            total += self.current_memory_dictionary[i].weight

        for each_memory in selected_distance.keys():
            prob_density += self.current_memory_dictionary[each_memory].weight/total*epanechnikov_kernel(selected_distance[each_memory],bandwidth)
        return prob_density

    def find_bandwidth(self,data_dictionary):
        total = 0
        std = 0
        for each_memory in data_dictionary:
            total += np.array(self.current_memory_dictionary[each_memory].data_point.carla_tensor.detach().cpu()) * self.current_memory_dictionary[each_memory].weight
        mean = total / len(data_dictionary)
        for each_memory in data_dictionary:
            std += np.square((np.array(self.current_memory_dictionary[each_memory].data_point.carla_tensor.detach().cpu()) - mean))* self.current_memory_dictionary[each_memory].weight
        std = np.sqrt(std)
        bandwidth = 1.06*std*(len(data_dictionary)**())
        return bandwidth
