from scripts.martingales import SMM
from scripts.detector import StatefulDetector, StatelessDetector
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='Out-of-distribution detection offline.')
parser.add_argument("-o", "--out", help="set test images from the out-of-distribution test set", action="store_true", default=False)
parser.add_argument("-v", "--vae", help="set nonconformity measure as VAE-based method", action="store_true", default=False)
parser.add_argument("-N", help="sliding window size for SVDD; # of generated examples for VAE", type=int, default=10)
args = parser.parse_args()


# load the test images
if args.out:
    data_path = "./data/out"
else:
    data_path = "./data/in"
test_images = np.load(os.path.join(data_path, "X.npy"))

if args.vae:
    from scripts.icad_vae import ICAD_VAE
    icad = ICAD_VAE()
    detector = StatefulDetector(sigma=6, tau=156)
    for idx, test_image in enumerate(test_images):
        smm = SMM(args.N)
        for i in range(args.N):
            p = icad(test_image)
            m = np.log(smm(p))
        S, d = detector(m)
        print("Time step: {}\t p-value: {}\t logM: {}\t S: {}\t Detector: {}".format(idx, round(p,3), round(m, 3), round(S, 3), d))


else:
    from scripts.icad_svdd import ICAD_SVDD

    icad = ICAD_SVDD()
    smm = SMM(args.N)
    detector = StatelessDetector(tau=14)
    for idx, test_image in enumerate(test_images):
        p = icad(test_image)
        m = np.log(smm(p))
        d = detector(m)
        print("Time step: {}\t p-value: {}\t logM: {}\t Detector: {}".format(idx, round(p,3), round(m, 3), d))