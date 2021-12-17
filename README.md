Setup Vanderbilt_data in following folder structure at ./
```bash
├── Vanderbilt_data
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

# OURs



# NTU 

Also setup drift_data in following folder structure at ./
```bash
├── drift_data
│   ├── testing
│   │   ├── in
│   │   ├── out
│   ├── training
```

Run the following to extract features to ./NTU_features_all/. The following also produces the dictionary of frame lengths (frame_lens) that is pasted into the training and test code below.
```
python feature_abstraction_<name of dataset>.py
```

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