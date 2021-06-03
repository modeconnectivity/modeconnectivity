#!/bin/sh

DATAROOT="data/"
OUTROOT="results/"
DEPTH=5
WIDTH=500
VAR=3  # variation, number of models to train
LR=0.0005
# the main script for learning the different dropout models

#1. first learn the original models
for V in `seq 1 $VAR`; do  # can be performed in parallel

    python train_fcn.py --dataset cifar10 --dataroot $DATAROOT -oroot $OUTROOT/cifar10/fcn/ --name var-$V --depth $DEPTH --width $WIDTH --learning_rate $LR --nepoch 200 
    MODEL=$OUTROOT/cifar10/fcn/var-$V/checkpoint.pth

#2. once done, train all the subnetworks on the task
    for EL in `seq 0 $DEPTH`; do  # can be performed in parallel
        (( $EL == 0 )) && LRA=0.001 || LRA=0.005;
        python exp_a_fcn.py --model $MODEL  --nepoch 400 --fraction 2 --name A --ndraw 20 --entry_layer $EL --learning_rate $LRA
    done;


#2b. merge the results from the different layers
    python merge_a_fcn.py $OUTROOT/cifar10/fcn/

#3. Perform experiment B on the same network
    python exp_b.py --model $MODEL  --fraction 2 --name B --ndraw 200

done;

#4. Merge the different runs
python merge_vars.py $OUTROOT/cifar10/fcn/var-*

#5. Plot the 
python plot_merge.py $OUTROOT/cifar10/fcn/merge/ --yscale linear 
python plot_merge.py $OUTROOT/cifar10/fcn/merge/ --yscale log 

# plot the path
python path.py --nameA 'A' --nameB 'B' --M1 $OUTROOT/cifar10/fcn/var-1/checkpoint.pth --M2 $OUTROOT/cifar10/fcn/var-2/checkpoint.pth
python plot_path.py --file $OUTROOT/cifar10/fcn/path/path.csv


