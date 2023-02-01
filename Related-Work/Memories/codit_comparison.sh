#!/bin/bash

python3 main.py --predict_carla True --memory_dir ./memories/carla_memories_10_0.2  --initial_memory_threshold 0.2 --test_carla_dir ./CARLA_dataset/testing/in --prob_threshold 0.948 
--window_size 16 --window_threshold 14 --task in

python3 main.py --predict_carla True --memory_dir ./memories/carla_memories_10_0.2  --initial_memory_threshold 0.2 --test_carla_dir ./CARLA_dataset/testing/out_snowy --prob_threshold 0.948  
--window_size 16 --window_threshold 14 --task out_snowy

python3 main.py --predict_carla True --memory_dir ./memories/carla_memories_10_0.2  --initial_memory_threshold 0.2 --test_carla_dir ./CARLA_dataset/testing/out_rainy --prob_threshold 0.948  
--window_size 16 --window_threshold 14 --task out_rainy

python3 main.py --predict_carla True --memory_dir ./memories/carla_memories_10_0.2  --initial_memory_threshold 0.2 --test_carla_dir ./CARLA_dataset/testing/out_replay --prob_threshold 0.948 
--window_size 16 --window_threshold 14 --task out_replay

python3 main.py --predict_carla True --memory_dir ./memories/carla_memories_10_0.2  --initial_memory_threshold 0.2 --test_carla_dir ./CARLA_dataset/testing/out_night --prob_threshold 0.948 
--window_size 16 --window_threshold 14 --task out_night

python3 main.py --predict_carla True --memory_dir ./memories/carla_memories_10_0.2  --initial_memory_threshold 0.2 --test_carla_dir ./CARLA_dataset/testing/out_foggy --prob_threshold 0.948  
--window_size 16 --window_threshold 14 --task out_foggy