# This repo contains code for our paper, CODiT: Conformal Out-of-Distribution Detection in Time-series Data

# OURs
  cd ours and follow instructions in README.md for reproducing CODiT's results reported in the paper

# Related Work
Setup carla_data in following folder structure at ./
```bash
├── carla_data
│   ├── testing
│   │   ├── in
│   │   ├── out_foggy/out
│   │   ├── out_night/out
│   │   ├── out_rainy/out
│   │   ├── out_snowy/out
│   ├── training
│   │   ├── setting_1
│   │   ├── setting_2
│   │   ├── setting_3
```

# Feng et al.'s 

Also setup drift_data in following folder structure at ./
```bash
├── drift_data
│   ├── testing
│   │   ├── in
│   │   ├── out
│   ├── training
│   ├── calibration
```

Run the following to extract features to ./carla_features_all/. The following also produces the dictionary of frame lengths (frame_lens) that is pasted into the training and test code below.
```
python feature_abstraction_<name of dataset>.py
```
(where <name of dataset> can be 'carla' or 'drift')

Run code to train a model which will be saved within NTU directory.
```
cd NTU
python train_<name of dataset>.py
```

Test as follows. Plots will be saved to NTU/plots/ and results will be printed to stdout.
```
cd NTU
python test_<name of dataset>.py
```

# Cai et al.'s

To train 
```
cd Vanderbilt
mkdir carla_models
python train_vae_svdd.py -d carla
```

To test with VAE model
```
cd Vanderbilt
python detect.py -d carla -v
```

# Ramankrishna et al's

To train 
```
cd Beta-VAE
mkdir carla_models
cd bvae-train-test
python train-bvae_carla.py
```

To test with VAE model
```
cd Beta-VAE/bvae-train-test
python test-bvae_carla.py
```
