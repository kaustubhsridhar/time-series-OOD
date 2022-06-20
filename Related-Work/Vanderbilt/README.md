# Real-time Out-of-distribution Detector
This repository provides the implementation of the real-time out-of-distribution detector in advanced emergency braking system (AEBS) for our "Real-time Out-of-distribution Detection in Learning-Enabled Cyber-Physical Systems" ICCPS 2020 paper. 

## Paper
If you use our method, please cite our ICCPS'20 paper.

_Real-time Out-of-distribution Detection in Learning-Enabled Cyber-Physical-Systems_<br>Feiyang Cai and Xenofon Koutsoukos
[[PDF](https://arxiv.org/pdf/2001.10494.pdf)]
[[talk](https://www.youtube.com/watch?v=Lg7XQdo_JSA&feature=youtu.be)]
```
@article{cai2020real,
  title={Real-time Out-of-distribution Detection in Learning-Enabled Cyber-Physical Systems},
  author={Cai, Feiyang and Koutsoukos, Xenofon},
  journal={arXiv preprint arXiv:2001.10494},
  year={2020}
}
```

## Getting Started

For now, this repository only provides the codes to run the out-of-distribution detection method offline. The online detection codes in AEBS will be added in the future.

### Advanced Emergency Braking System

TBD

### Installing

The codes are written in `Python 3.7` and requires the packages listed in `requirement.txt`.

```
pip install -r requirements.txt
```

## Running the tests


### VAE and SVDD training
Here is the example to train the VAE and SVDD based detectors. (The traininig data set will be added later)

```
python ./train_vae_svdd.py -p ./data/train
```

### Offline out-of-distribution detection

We provide one episode of in-distribution data and one episode of out-of-distribution data to test our method. 

Please have a look into `detect.py` for possible arguments to set test data and detection method (VAE or SVDD).

Here is the example to run the SVDD-based detection method for the out-of-distribution data:
```
python ./detect.py -v -o
```

# License
MIT