#!/bin/sh

DATAROOT="data/"
OUTROOT="results/"
DEPTH=2
WIDTH=500
VAR=3  # variation, number of models to train
LR=0.0005

for V in `seq 1 $VAR`; do
    for W in `seq 50 500 50` `seq 600 1500 100` `seq 1800 3000 300`; do
        python train_fcn.py --dataset cifar10 --dataroot $DATAROOT -oroot $OUTROOT/cifar10/vary_width/ --name W-$W/var-$V --depth $DEPTH --width $WIDTH --learning_rate $LR --nepoch 200 

        MODEL=$OUTROOT/cifar10/vary_width/W-$W/var-$V/checkpoint.pth

        for EL in `seq 0 $DEPTH`; do
            (( $EL == 0 )) && lr=0.001 || lr=0.005
            python exp_a_fcn.py --model $MODEL  --nepoch 400 --fraction 2 --name A --ndraw 20 --entry_layer $EL
        done;

        python merge_a_fcn.py $OUTROOT/cifar10/vary_width/W-$W

        python exp_b.py --model $MODEL  --fraction 2 --name B --ndraw 200
    done;
    python merge_max_widths.py $OUTROOT/cifar10/vary_width/W-*
done;

python plot_widths.py $OUTROOT/cifar10/vary_width/merge/


