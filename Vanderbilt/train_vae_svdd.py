import numpy as np
from scripts.vae_svdd_trainer import Trainer
import os
import argparse


parser = argparse.ArgumentParser(description='Train the detector.')
# parser.add_argument("-p", "--path", help="set the training data path")
parser.add_argument("-d", "--data", help="choose from [carla, drift]")
args = parser.parse_args()

if not args.data:
    print("Please provide data type (one of [carla, drift])")
else:
    # X_train = np.load(os.path.join(args.path, "X.npy"))
    # y_train = np.load(os.path.join(args.path, "Y.npy"))
    trainer = Trainer(args.data)
    trainer.fit()