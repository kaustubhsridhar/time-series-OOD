rm -f ./results/carla_oods_bike_exp_results.txt

python3 main.py --predict_carla True --memory_dir ./memories/carla_memories_10_0.2  --test_carla_dir ./test_carla/oods_bike \
--prob_threshold 0.92 --window_size 5 --window_threshold 5 --task oods_bike

echo "finish bike experiment 1/4"

python3 main.py --predict_carla True --memory_dir ./memories/carla_memories_10_0.2  --test_carla_dir ./test_carla/oods_bike \
--prob_threshold 0.92 --window_size 10 --window_threshold 5 --task oods_bike

echo "finish bike experiment 2/4"

python3 main.py --predict_carla True --memory_dir ./memories/carla_memories_10_0.3  --test_carla_dir ./test_carla/oods_bike \
--prob_threshold 0.78 --window_size 5 --window_threshold 5 --task oods_bike

echo "finish bike experiment 3/4"

python3 main.py --predict_carla True --memory_dir ./memories/carla_memories_10_0.3  --test_carla_dir ./test_carla/oods_bike \
--prob_threshold 0.78 --window_size 10 --window_threshold 5 --task oods_bike

echo "finish bike experiment 4/4"