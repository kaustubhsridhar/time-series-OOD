## This code is build on top of
      VCOP - https://github.com/xudejing/video-clip-order-prediction
      Vandebilt - 
      
## Create conda environment and install requirements
      conda create --name codit python=3.6.13
      conda activate codit
      cd ours
      pip install -r requirements.txt

## Download trained models: https://drive.google.com/drive/folders/1xBIkVB7TpIcRJPfF3hxzybe5KVPdbFrF?usp=sharing 
      unzip saved_models

## Download data: https://drive.google.com/drive/folders/1JjHKjANN5W6Y_pUmX5tOBIQcEUsqbErf?usp=sharing
      unzip data
     
## Download pre-computed fisher and p-values
      CARLA: https://drive.google.com/drive/folders/1lkejBox1MWdMnNjmZU_JdYd1y1XYml0M?usp=sharing
      unzip carla_log
      DRIFT: 
      unzip
      
## Generate results for Weather OODs (Table 2)
      Rainy: python check_carla.py --gpu $0/1/2/3$ --cuda --ckpt saved_models/carla_model.pt --n 20 --out_folder_name out_rainy/out --save_dir carla_log/rainy --transformation_list speed shuffle reverse periodic identity
      Foggy: python check_carla.py --gpu $0/1/2/3$ --cuda --ckpt saved_models/carla_model.pt --n 20 --out_folder_name out_foggy/out --save_dir carla_log/foggy --transformation_list speed shuffle reverse periodic identity
      Snowy: python check_carla.py --gpu $0/1/2/3$ --cuda --ckpt saved_models/carla_model.pt --n 20 --out_folder_name out_snowy/out --save_dir carla_log/snowy --transformation_list speed shuffle reverse periodic identity
      Night: python check_carla.py --gpu $0/1/2/3$ --cuda --ckpt saved_models/carla_model.pt --n 20 --out_folder_name out_night/out --save_dir carla_log/night --transformation_list speed shuffle reverse periodic identity
     
## Generate results for Replay OODs (Figure 7 (left))
      python check_carla.py --gpu $0/1/2/3$ --cuda --ckpt saved_models/carla_model.pt --n 20 --out_folder_name out_replay/out --save_dir carla_log/replay --transformation_list speed shuffle reverse periodic identity

