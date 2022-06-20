# python train_drift.py --nd 18 --model_save_folder ./drift_models_cl_18
# python train_drift.py --nd 20 --model_save_folder ./drift_models_cl_20
# python train_drift.py --nd 22 --model_save_folder ./drift_models_cl_22
# python train_drift.py --nd 24 --model_save_folder ./drift_models_cl_24


python test_drift.py --nd 18 --model_save_folder ./drift_models_cl_18
python test_drift.py --nd 20 --model_save_folder ./drift_models_cl_20
python test_drift.py --nd 22 --model_save_folder ./drift_models_cl_22
python test_drift.py --nd 24 --model_save_folder ./drift_models_cl_24