#!/bin/sh

DATAROOT="data/"
OUTROOT="results/mnist/fcn"
DEPTH=10
WIDTH=245
LR=0.001
VAR=10  # variation, number of models to train
F=2  # the dropout ratio denominator
# the main script for learning the different dropout models

for V in `seq 1 $VAR`; do  # can be performed in parallel

    #1. first learn the original models
    python train_fcn.py --dataset mnist --dataroot $DATAROOT -oroot $OUTROOT --name var-$V --depth $DEPTH --width $WIDTH --learning_rate $LR --nepoch 200 
    MODEL=$OUTROOT/var-$V/checkpoint.pth

#2. once done, train all the subnetworks on the task
    for EL in `seq 0 $DEPTH`; do  # can be performed in parallel
        LRA=0.003
        python exp_a_fcn.py --model $MODEL  --nepoch 400 --fraction $F --name "A-f$F" --ndraw 20 --entry_layer $EL --learning_rate $LRA
    done;


#2b. merge the results from the different layers
    python merge_a_fcn.py $OUTROOT

#3. Perform experiment B on the same network
    python exp_b.py --model $MODEL  --fraction $F --name "B-f$F" --ndraw 200

done;

#4. Merge the different runs
python merge_vars.py $OUTROOT/var-*

#5. Plot the different runs
python plot_merge.py $OUTROOT/merge/ --yscale linear --experiments "A-f$F" "B-f$F"
python plot_merge.py $OUTROOT/merge/ --yscale log    --experiments "A-f$F" "B-f$F"

