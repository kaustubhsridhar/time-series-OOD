# These are the instruction for reproducing CODiT's results on CARLA and DRIFT datasets. For generating results on GAIT dataset, cd gait and follow instructions in gait/README.md

## This code is build on
     1. Video CLip Order Prediction's code - https://github.com/xudejing/video-clip-order-prediction
     2. Cai et al.'s for CARLA dataset - https://github.com/feiyang-cai/out_of_distribution_detector_aebs 
      
## Create conda environment and install requirements
      conda create --name codit python=3.6.13
      conda activate codit
      cd ours
      pip install -r requirements.txt

## Download trained models: https://drive.google.com/drive/folders/1xBIkVB7TpIcRJPfF3hxzybe5KVPdbFrF?usp=sharing 
      mkdir saved_models
      mv carla.pt and drift.pt to ./saved_models

## Download data: https://drive.google.com/file/d/11iJF2UQx4z78hfC8C9SFgapMgl_17SaJ/view?usp=sharing
     unzip data.zip
      
## Download pre-computed fisher and p-values: https://drive.google.com/drive/folders/1o2bQ6M17kvN6b78KYPuAv0oavZ0Mf926?usp=sharing
      CARLA: unzip carla_log
      Drift: unzip drift_log
      
## Generate results for Weather OODs (Table 2)
      Rainy: python check_OOD_carla.py --gpu $0/1/2/3$ --cuda --ckpt saved_models/carla_model.pt --n 20 --out_folder_name out_rainy/out --save_dir carla_log/rainy --transformation_list speed shuffle reverse periodic identity
      Foggy: python check_OOD_carla.py --gpu $0/1/2/3$ --cuda --ckpt saved_models/carla_model.pt --n 20 --out_folder_name out_foggy/out --save_dir carla_log/foggy --transformation_list speed shuffle reverse periodic identity
      Snowy: python check_OOD_carla.py --gpu $0/1/2/3$ --cuda --ckpt saved_models/carla_model.pt --n 20 --out_folder_name out_snowy/out --save_dir carla_log/snowy --transformation_list speed shuffle reverse periodic identity
      Night: python check_OOD_carla.py --gpu $0/1/2/3$ --cuda --ckpt saved_models/carla_model.pt --n 20 --out_folder_name out_night/out --save_dir carla_log/night --transformation_list speed shuffle reverse periodic identity
     
## Generate results for Replay OODs (Figure 7 (left))
      python check_OOD_carla.py --gpu $0/1/2/3$ --cuda --ckpt saved_models/carla_model.pt --n 20 --out_folder_name out_replay/out --save_dir carla_log/replay --transformation_list speed shuffle reverse periodic identity

## Generate results for Drift OODs (Figure 7 (left))
      python check_OOD_drift.py --gpu $0/1/2/3$ --cuda --ckpt saved_models/drift.pt --n 20 --save_dir drift_log --transformation_list speed shuffle reverse periodic identity


### (optional) Train VAE model for precition of the applied transformation on the CARLA dataset
     python train_carla.py --cl 16 --log saved_models --bs 2 --gpu 0 --transformation_list speed shuffle reverse periodic identity
### (optional) Train VAE model for precition of the applied transformation on the drift dataset
     python train_drift.py --cl 16 --log saved_models --bs 2 --gpu 0 --lr 0.00001


