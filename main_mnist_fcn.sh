#!/bin/sh

DATAROOT="data/"
OUTROOT="results/"
DEPTH=10
WIDTH=245
# the main script for learning the different dropout models

#1. first learn the original model
python train_fcn.py --dataset mnist --dataroot $DATAROOT -oroot $OUTROOT --name mnist-fcn --depth $DEPTH --width $WIDTH --nepoch 200


#2. once done, train all the subnetworks on the task
for el in `seq 0 $DEPTH`; do
    python exp_a_fcn.py --model $OUTROOT/mnist-fcn/checkpoint.pth  --nepoch 1000 --fraction 2 --name A --ndraw 20 --entry_layer $el
done


#2b. merge the results from the different layers
python merge_a_fcn.py $OUTROOT/mnist-fcn/

#3. Perform experiment B on the same network
python exp_b.py --model $OUTROOT/mnist-fcn/checkpoint.pth --fraction 2 --name B --ndraw 200

#4 plot the two
python plot_meta.py $OUTROOT/mnist-fcn/ --experiments A B


