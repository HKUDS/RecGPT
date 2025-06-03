#!/bin/bash

nohup accelerate launch pre_train.py --batch_size 40 --epoch 5 --tf_layer 3 > ./logs/layer_3.log 2>&1