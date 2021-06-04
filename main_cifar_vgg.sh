#!/bin/sh

DATAROOT="data/"
OUTROOT="results/cifar10/vgg"
VAR=5  # variation, number of models to train
F=2
LR=0.005
# the main script for learning the different dropout models

for V in `seq 1 $VAR`; do  # can be performed in parallel
    #1. first learn the original model
    python train_vgg.py --dataset cifar10 --dataroot $DATAROOT -oroot $OUTROOT --name var-$V --model vgg-11 --nepoch 200 -lr $LR 

    MODEL=$OUTROOT/var-$V/checkpoint.pth

#2. once done, train all the subnetworks on the task
    for EL in `seq 0 10`; do
        LRA=0.005
        OPT=""
        if [ "$F" -eq 4 ]; then  # trickier to train with F=4
            if [ "$EL" -eq 0 ]; then
            OPT="-lrg 0.5 -lrs 250"
            fi
            if [ "$EL" -eq 1 ]; then
                OPT="-lrg 0.9 -lrs -1"
            fi
        fi
        python exp_a_vgg.py --model $MODEL --nepoch 200 --fraction $F --name "A-f$F" --ndraw 20 --entry_layer $EL -lr $LRA  $OPT &
    done


#2b. merge the results from the different layers
    python merge_a_vgg.py $OUTROOT

#3. Perform experiment B on the same network
    python exp_b.py --model $MODEL --fraction $F --name "B-f$F" --ndraw 200

done;
#4. Merge the different runs
python merge_vars.py $OUTROOT/var-*

# 5. Plot the different runs
python plot_merge.py $OUTROOT/merge/ --yscale linear  --experiments "A-f$F" "B-f$F"
python plot_merge.py $OUTROOT/merge/ --yscale log  --experiments "A-f$F" "B-f$F"

