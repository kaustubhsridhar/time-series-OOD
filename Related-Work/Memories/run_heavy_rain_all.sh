echo "Warning: running this script takes about 7 hour"

dist=0.2

for win_size in 5 10 15
do
    for prob in 0.8 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.0 1.01 1.02 1.03 1.04 1.05
    do 
        python3 main.py --predict_carla True --memory_dir ./memories/carla_memories_10_"$dist"  --initial_memory_threshold $dist --test_carla_dir ./test_carla \
        --prob_threshold $prob --window_size $win_size --window_threshold 5 --task heavy_rain

    done
done
