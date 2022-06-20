## train time

# nohup python -u train_vae_svdd.py -d carla > carla_models/carla_train.log &

# nohup python -u train_vae_svdd.py -d drift > drift_models/drift_train.log &

## test time

# nohup python -u detect.py -d carla -v > carla_models/carla_vae_test.log &

# nohup python -u detect.py -d carla > carla_models/carla_svdd_test.log &


# nohup python -u detect.py -d drift -v > drift_models/drift_vae_test.log &

# nohup python -u detect.py -d drift > drift_models/drift_svdd_test.log &

# python detect.py -d carla -v
nohup python -u detect.py -d carla -v > carla_models/carla_vae_test_out_replay_only.log &
# python detect.py -d drift -v