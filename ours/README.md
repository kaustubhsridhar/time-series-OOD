# These are instructions for reproducing CODiT's results on CARLA (Table 2) and DRIFT (Fig. 9 (left)) datasets on a Linux server with GPUs. 

For generating results on GAIT dataset, cd gait and follow instructions in gait/README.md

## This code is build on top of
     1. Video Clip Order Prediction's code - https://github.com/xudejing/video-clip-order-prediction
     2. Cai et al.'s for CARLA dataset - https://github.com/feiyang-cai/out_of_distribution_detector_aebs 

## Step 1: Setting up the environment: Two options here : (1) Docker, (2) Conda

### This step is for seting up the Docker environment.

1. install [Docker](https://docs.docker.com/get-docker/) on your machine 
2. To build a docker image: `docker build -t codit .` <br>
3. To run the docker container and open an interactive session with docker: `docker run -i -t --gpus all --name temp_test --rm codit /bin/bash`

After finishing the experiments, to leave the docker environment, 
run `exit` <br>

### Create conda environment and install requirements (It  takes ~20-30 minutes for installing requirements)
      conda create --name codit python=3.6.13
      conda activate codit
      pip install -r requirements.txt

## Step 2: Download trained models: https://drive.google.com/drive/folders/1xBIkVB7TpIcRJPfF3hxzybe5KVPdbFrF?usp=sharing 
      mkdir saved_models
      mv carla_model.pt and drift.pt to ./saved_models

## Step 3: Download data (zipped folder is ~13 GB, will take sometime to download): https://drive.google.com/file/d/11iJF2UQx4z78hfC8C9SFgapMgl_17SaJ/view?usp=sharing
     unzip data.zip
      
## Step 4: Download the two folders (carla_log, and drift_log) containing pre-computed fisher and p-values for 1 run for speedy evaluation: https://drive.google.com/drive/folders/1o2bQ6M17kvN6b78KYPuAv0oavZ0Mf926?usp=sharing
### Note: Rename the downloaded zip files to carla_log.zip and drift_log.zip respectively
      CARLA: unzip carla_log.zip
      Drift: unzip drift_log.zip
      
# Step 5: Generate the following results after populating --gpu command line argument with the gpu number (0/1/2/3) on which you are running these experiments

## Generate AUROC and detection delay results for Weather and Night OODs (Table 2)
### Note: The following results will be generated for just 1 run. In the paper, we ran these experiments 5 times (with different seeds) and reported the mean and standard deviation (std). For AUROC with low std (except for snowy), the results for 1 run are very close to the ones reported in paper. Similar results are for detection delay (except for night with high std). Here, we found that the detection delay for rainy to be 0.05 seconds higher (0.918) than the one reported in paper (0.86). We will rectify it in the final version.

      Rainy: python check_OOD_carla.py --gpu $0/1/2/3$ --cuda --ckpt saved_models/carla_model.pt --n 20 --out_folder_name out_rainy/out --save_dir carla_log/rainy --transformation_list speed shuffle reverse periodic identity
      Foggy: python check_OOD_carla.py --gpu $0/1/2/3$ --cuda --ckpt saved_models/carla_model.pt --n 20 --out_folder_name out_foggy/out --save_dir carla_log/foggy --transformation_list speed shuffle reverse periodic identity
      Snowy: python check_OOD_carla.py --gpu $0/1/2/3$ --cuda --ckpt saved_models/carla_model.pt --n 20 --out_folder_name out_snowy/out --save_dir carla_log/snowy --transformation_list speed shuffle reverse periodic identity
      Night: python check_OOD_carla.py --gpu $0/1/2/3$ --cuda --ckpt saved_models/carla_model.pt --n 20 --out_folder_name out_night/out --save_dir carla_log/night --transformation_list speed shuffle reverse periodic identity
     
## Generate AUROC and TNR results for Replay OODs (Figure 9 (left))
      python check_OOD_carla.py --gpu $0/1/2/3$ --cuda --ckpt saved_models/carla_model.pt --n 20 --out_folder_name out_replay/out --save_dir carla_log/replay --printTNR 1 --transformation_list speed shuffle reverse periodic identity

## Generate AUROC and TNR results for Drift OODs (Figure 9 (left))
      python check_OOD_drift.py --gpu $0/1/2/3$ --cuda --ckpt saved_models/drift.pt --n 20 --save_dir drift_log --transformation_list speed shuffle reverse periodic identity


### (optional) Train VAE model for precition of the applied transformation on the CARLA dataset
     python train_carla.py --cl 16 --log saved_models --bs 2 --gpu 0 --transformation_list speed shuffle reverse periodic identity
### (optional) Train VAE model for precition of the applied transformation on the drift dataset
     python train_drift.py --cl 16 --log saved_models --bs 2 --gpu 0 --lr 0.00001


