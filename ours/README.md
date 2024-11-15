# These are instructions for reproducing CODiT's results on CARLA (Table 2) and DRIFT (Fig. 9 (left)) datasets on a Linux server with GPUs. 

For generating results on GAIT dataset, cd gait and follow instructions in gait/README.md

## This code is built on top of
     1. Video Clip Order Prediction's code - https://github.com/xudejing/video-clip-order-prediction
     2. Cai et al.'s for CARLA dataset - https://github.com/feiyang-cai/out_of_distribution_detector_aebs 

## Step 1: Download trained models: https://drive.google.com/file/d/1H65NF7tcxA3XcKyh1C54nSdriM9QyFGu/view?usp=drive_link 
      mkdir saved_models
      mv carla_model.pt and drift.pt to ./saved_models

## Step 2: Download data (zipped folder is ~13 GB, will take sometime to download): 
https://drive.google.com/file/d/1PUoOJ_Oza3pkBFpcTCqa_GBjPw5JJjXU/view?usp=drive_link 
     unzip data.zip
      
## Step 3: Download the two folders (carla_log, and drift_log) containing pre-computed fisher and p-values for 1 run for speedy evaluation: https://drive.google.com/file/d/1GXtkM9CTXcPpLlhfcTORjg3xAEwc2XS0/view?usp=drive_link
### Note: Rename the downloaded zip files to carla_log.zip and drift_log.zip respectively
      CARLA: unzip carla_log.zip
      Drift: unzip drift_log.zip
      
## Step 4: Setting up the environment: Two options here : (1) Docker,or (2) Conda. You can work with either of these two options.

### This step is for seting up the Docker environment: takes ~50-60 minutes.

1. install [Docker](https://docs.docker.com/get-docker/) on your machine 
2. To build a docker image: `docker build -t codit .` <br>
3. To run the docker container and open an interactive session with docker: `docker run -i -t --gpus all --name temp_test --rm codit /bin/bash`

After finishing the experiments, to leave the docker environment, 
run `exit` <br>

### Create conda environment and install requirements (It  takes ~20-30 minutes for installing requirements)
      conda create --name codit python=3.6.13
      conda activate codit
      pip install -r requirements.txt
      
# Step 5: Generate the following results after populating --gpu command line argument with the gpu number (0/1/2/3) on which you are running these experiments

## Generate AUROC and detection delay results for Weather and Night OODs (Table 2)

The expected result, i.e., Table 2 from the paper is: <br>
<img src="images/Table2.png" width="800" />

### Note: The following results will be generated for just 1 run. In the paper, we ran these experiments 5 times (with different seeds) and reported the mean and standard deviation (std). For AUROC with low std (except for snowy), the results for 1 run are very close to the ones reported in paper. Similar results are for detection delay (except for night with high std). Here, we found that the detection delay for rainy to be 0.05 seconds higher (0.918) than the one reported in paper (0.86). We will rectify it in the final version.

      Rainy: python3 check_OOD_carla.py --gpu 0 --cuda --ckpt saved_models/carla_model.pt --n 20 --out_folder_name out_rainy/out --save_dir carla_log/rainy --transformation_list speed shuffle reverse periodic identity
      Foggy: python3 check_OOD_carla.py --gpu 0 --cuda --ckpt saved_models/carla_model.pt --n 20 --out_folder_name out_foggy/out --save_dir carla_log/foggy --transformation_list speed shuffle reverse periodic identity
      Snowy: python3 check_OOD_carla.py --gpu 0 --cuda --ckpt saved_models/carla_model.pt --n 20 --out_folder_name out_snowy/out --save_dir carla_log/snowy --transformation_list speed shuffle reverse periodic identity
      Night: python3 check_OOD_carla.py --gpu 0 --cuda --ckpt saved_models/carla_model.pt --n 20 --out_folder_name out_night/out --save_dir carla_log/night --transformation_list speed shuffle reverse periodic identity
     
## Generate AUROC and TNR results for Replay OODs (Figure 9 (left))

      python3 check_OOD_carla.py --gpu 0 --cuda --ckpt saved_models/carla_model.pt --n 20 --out_folder_name out_replay/out --save_dir carla_log/replay --printTNR 1 --transformation_list speed shuffle reverse periodic identity

Expected AUROC and TNR results are as shown in Figure 9 (left) from the paper: <br>
<img src="images/drift_n_replay.png" width="500" />

## Generate AUROC and TNR results for Drift OODs (Figure 9 (left))
      python3 check_OOD_drift.py --gpu 0 --cuda --ckpt saved_models/drift.pt --n 20 --save_dir drift_log --transformation_list speed shuffle reverse periodic identity


### (optional) Train VAE model for precition of the applied transformation on the CARLA dataset
     python3 train_carla.py --cl 16 --log saved_models --bs 2 --gpu 0 --transformation_list speed shuffle reverse periodic identity
### (optional) Train VAE model for precition of the applied transformation on the drift dataset
     python3 train_drift.py --cl 16 --log saved_models --bs 2 --gpu 0 --lr 0.00001


