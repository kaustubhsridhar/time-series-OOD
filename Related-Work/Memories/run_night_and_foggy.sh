rm -f ./results/carla_oods_foggy_exp_results.txt
rm -f ./results/carla_oods_night_exp_results.txt

echo "run night exps"

python3 main.py --predict_carla True --memory_dir ./memories/carla_memories_10_0.2  --initial_memory_threshold 0.2 --test_carla_dir ./test_carla/oods_night \
--prob_threshold 0.92 --window_size 5 --window_threshold 5 --task oods_night

echo "finish night experiment 1/4"

python3 main.py --predict_carla True --memory_dir ./memories/carla_memories_10_0.2  --initial_memory_threshold 0.2 --test_carla_dir ./test_carla/oods_night \
--prob_threshold 0.92 --window_size 10 --window_threshold 5 --task oods_night

echo "finish night experiment 2/4"

python3 main.py --predict_carla True --memory_dir ./memories/carla_memories_10_0.3 --initial_memory_threshold 0.3 --test_carla_dir ./test_carla/oods_night \
--prob_threshold 0.78 --window_size 5 --window_threshold 5 --task oods_night

echo "finish night experiment 3/4"

python3 main.py --predict_carla True --memory_dir ./memories/carla_memories_10_0.3  --initial_memory_threshold 0.3 --test_carla_dir ./test_carla/oods_night \
--prob_threshold 0.78 --window_size 10 --window_threshold 5 --task oods_night

echo "finish night experiment 4/4"

echo "run foggy exps"

python3 main.py --predict_carla True --memory_dir ./memories/carla_memories_10_0.2  --initial_memory_threshold 0.2 --test_carla_dir ./test_carla/oods_foggy \
--prob_threshold 0.92 --window_size 5 --window_threshold 5 --task oods_foggy

echo "finish foggy experiment 1/4"

python3 main.py --predict_carla True --memory_dir ./memories/carla_memories_10_0.2  --initial_memory_threshold 0.2 --test_carla_dir ./test_carla/oods_foggy \
--prob_threshold 0.92 --window_size 10 --window_threshold 5 --task oods_foggy

echo "finish foggy experiment 2/4"

python3 main.py --predict_carla True --memory_dir ./memories/carla_memories_10_0.3  --initial_memory_threshold 0.3 --test_carla_dir ./test_carla/oods_foggy \
--prob_threshold 0.78 --window_size 5 --window_threshold 5 --task oods_foggy

echo "finish foggy experiment 3/4"

python3 main.py --predict_carla True --memory_dir ./memories/carla_memories_10_0.3  --initial_memory_threshold 0.3 --test_carla_dir ./test_carla/oods_foggy \
--prob_threshold 0.78 --window_size 10 --window_threshold 5 --task oods_foggy

echo "finish foggy experiment 4/4"

echo "finish both experiments"