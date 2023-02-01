rm -f ./results/carla_heavy_rain_exp_results.txt

python3 main.py --predict_carla True --memory_dir ./memories/carla_memories_10_0.2  --initial_memory_threshold 0.2 --test_carla_dir ./test_carla \
--prob_threshold 0.92 --window_size 5 --window_threshold 5 --task heavy_rain

echo "finish heavy rain experiment 1/4"

python3 main.py --predict_carla True --memory_dir ./memories/carla_memories_10_0.2  --initial_memory_threshold 0.2 --test_carla_dir ./test_carla \
--prob_threshold 0.92 --window_size 10 --window_threshold 5 --task heavy_rain

echo "finish heavy rain experiment 2/4"

python3 main.py --predict_carla True --memory_dir ./memories/carla_memories_10_0.3   --initial_memory_threshold 0.3 --test_carla_dir ./test_carla \
--prob_threshold 0.78 --window_size 5 --window_threshold 5 --task heavy_rain

echo "finish heavy rain experiment 3/4"

python3 main.py --predict_carla True --memory_dir ./memories/carla_memories_10_0.3   --initial_memory_threshold 0.3 --test_carla_dir ./test_carla \
--prob_threshold 0.78 --window_size 10 --window_threshold 5 --task heavy_rain

echo "finish heavy rain experiment 4/4"