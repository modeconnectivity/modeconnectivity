#!/bin/sh

DATAROOT="data/"
OUTROOT="results/"
DEPTH=2
WIDTH=245
# the main script for learning the different dropout models

#1. first learn the original model
#python train_fcn.py --dataset cifar10 --dataroot $DATAROOT -oroot $OUTROOT --name cifar10-fcn --depth $DEPTH --width $WIDTH --nepoch 2 

#2. once done, train all the subnetworks on the task
#for el in `seq 0 $DEPTH`; do
    #python exp_a_fcn.py --model $OUTROOT/cifar10-fcn/checkpoint.pth  --nepoch 2 --fraction 2 --name A --ndraw 2 --entry_layer $el
#done


#2b. merge the results from the different layers
#python merge_a_fcn.py $OUTROOT/cifar10-fcn/

#3. Perform experiment B on the same network
python exp_b.py --model $OUTROOT/cifar10-fcn/checkpoint.pth --fraction 2 --name B --ndraw 2

#4 plot the two
python plot_meta.py $OUTROOT/cifar10-fcn/ --experiments A B


