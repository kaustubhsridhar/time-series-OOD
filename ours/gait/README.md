## Step 1: Install requirements
    conda create --name gait python=3.6.13
    conda activate gait
    pip install -r requirement.txt

## Step 2: Download gait-in-neurodegenerative-disease-database-1.0.0 folder from https://drive.google.com/drive/folders/1Z-3YnlhcCxI_KlFF6FF7tMp5MSEZURRH?usp=sharing
### Note: remave the downloaded folder to gait-in-neurodegenerative-disease-database-1.0.0.zip
    mkdir data
    cd data
    mv gait-in-neurodegenerative-disease-database-1.0.0.zip .
    unzip gait-in-neurodegenerative-disease-database-1.0.0.zip
    cd ../

## Step 3: Download the three trained models (gait_16.pt, gait_18.pt, gait_20.pt) from https://drive.google.com/drive/folders/1p0F2D3oTUgB3QKq_0uLu1F9eKRzRibml?usp=sharing
    mkdir saved_models
    mv gait_16.pt saved_models/.
    mv gait_18.pt saved_models/.
    mv gait_20.pt saved_models/.

## Step 4: Generate CODiT results in Table 4 with $wl$=16/18/20, $disease@_type$=als/park/hunt/all
    mkdir gait_log
    python check_OOD_gait.py --save_dir gait_log/ --ckpt saved_models/gait_$wl$.pt  --transformation_list high_pass low_high high_low identity --wl $wl$ --cuda --gpu 0 --n 100 --disease_type $disease_type$

## (optional) Training VAE model on GAIT dataset on $wl$=16/18/20
    python train_gait.py --log saved_models --transformation_list high_pass low_high high_low identity --wl $wl$
    
## Generating baseline results with $wl$=16/18/20, $disese@_type$ als/hunt/park/all in Table 4
    python check_OOD_baseline.py --disease_type $disease_type$ --wl $wl$ --root_dir data/gait-in-neurodegenerative-disease-database-1.0.0

