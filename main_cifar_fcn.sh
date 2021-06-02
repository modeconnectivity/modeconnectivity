#!/bin/sh

DATAROOT="data/"
OUTROOT="results/"
DEPTH=5
WIDTH=500
VAR=3  # variation, number of models to train
LR=0.0005
# the main script for learning the different dropout models

#1. first learn the original models
for V in `seq 1 $VAR`; do

    python train_fcn.py --dataset cifar10 --dataroot $DATAROOT -oroot $OUTROOT/cifar10/fcn/ --name var-$V --depth $DEPTH --width $WIDTH --learning_rate $LR --nepoch 200 
    MODEL=$OUTROOT/cifar10/fcn/var-$V/checkpoint.pth

#2. once done, train all the subnetworks on the task
    for EL in `seq 0 $DEPTH`; do
        (( $EL == 0 )) && lr=0.001 || lr=0.005
        python exp_a_fcn.py --model $MODEL  --nepoch 400 --fraction 2 --name A --ndraw 20 --entry_layer $EL
    done;


#2b. merge the results from the different layers
    python merge_a_fcn.py $OUTROOT/cifar10/fcn/

#3. Perform experiment B on the same network
    python exp_b.py --model $MODEL  --fraction 2 --name B --ndraw 200

#4 plot the two
    #python plot_meta.py $OUTROOT/cifar10-fcn/ --experiments A B
done;

python merge_vars.py $OUTROOT/cifar10/fcn/var-*
python plot_merge.py $OUTROOT/cifar10/fcn/merge/ --experiments A B

#python path.py -A $OUTROOT/cifar10/fcn/ -B $OUTROOT/cifar10/fcn


