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