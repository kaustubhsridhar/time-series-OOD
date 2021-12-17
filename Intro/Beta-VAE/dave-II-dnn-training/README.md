# LEC training and validation
These scripts are used to train an end-to-end learning DNN to steer the autonomous vehicle. We use the NVIDIA's DAVE II DNN model. 

```
python3 train.py                       --- train the DNN using the data generated.

python3 test.py                        --- test the DNN's prediction.

python3 performance-calculator.py      --- Measure the performance of the trained model using mean square error.
```
