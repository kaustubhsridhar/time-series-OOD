# The following instructions for generating results (Table 4) on GAIT dataset. This code has been tested on a server with GPUs.

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

## Step 4: Generate CODiT results in Table 4 after populating the command-line arguments: --ckpt as saved_models/gait_$wl$.pt where wl = 16/18/20, --wl=16/18/20 (same as wl in saved_models/gait_$wl$.pt), and --disease\_type=als/park/hunt/all
### park is for PD, hunt is for HD

### Note: The following results will be generated for just 1 run. In the paper, we ran these experiments 5 times (with different seeds) and reported the mean and standard deviation (std). The results with 1 run are similar to those reported in paper. 

### It takes ~10-15 minutes to run for "all", as OOD data, and ~ minutes for "als/park/hunt" as OOD data.

    mkdir gait_log
    python check_OOD_gait.py --save_dir gait_log/ --ckpt saved_models/gait_$wl$.pt  --transformation_list high_pass low_high high_low identity --wl $wl$ --cuda --gpu 0 --n 100 --disease_type $disease_type$
    
## Generate baseline results with --wl=16/18/20, --disese\_type als/hunt/park/all in Table 4
    python check_OOD_baseline.py --disease_type $disease_type$ --wl $wl$ --root_dir data/gait-in-neurodegenerative-disease-database-1.0.0

## (optional) Training VAE model on GAIT dataset on wl=16/18/20
    python train_gait.py --log saved_models --transformation_list high_pass low_high high_low identity --wl $wl$

