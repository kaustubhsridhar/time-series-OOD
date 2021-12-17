import numpy as np
from scripts.vae_svdd_trainer import Trainer
import os
import argparse


parser = argparse.ArgumentParser(description='Train the detector.')
parser.add_argument("-p", "--path", help="set the training data path")
args = parser.parse_args()

if not args.path:
    print("Please provide the training data path")
else:
    X_train = np.load(os.path.join(args.path, "X.npy"))
    y_train = np.load(os.path.join(args.path, "Y.npy"))

    trainer = Trainer(X_train, y_train)
    trainer.fit()