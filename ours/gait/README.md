## Install requirements
    conda create --name gait python=3.6.13
    conda activate gait
    cd gait
    pip install -r requirements.txt

## Download data from https://drive.google.com/drive/folders/1Z-3YnlhcCxI_KlFF6FF7tMp5MSEZURRH?usp=sharing
    mkdir data
    cd data
    unzip gait-in-neurodegenerative-disease-database-1.0.0.zip
    cd ../

## Download trained models from https://drive.google.com/drive/folders/1p0F2D3oTUgB3QKq_0uLu1F9eKRzRibml?usp=sharing
    mkdir saved_models
    mv gait_16.pt saved_models/.
    mv gait_18.pt saved_models/.
    mv gait_20.pt saved_models/.

## Check OOD with $wl$=16/18/20, $disease_type$=als/park/hunt/all
    mkdir gait_log
    python check_OOD_gait.py --save_dir gait_log/ --ckpt saved_models/gait_$wl$.pt  --transformation_list high_pass low_high high_low identity --wl $wl$ --cuda --gpu 0 --n 100 --disease_type $disease_type$

