#!/bin/sh

DATAROOT="data/"
OUTROOT="results/cifar10/fcn"
DEPTH=5
WIDTH=500
VAR=10  # variation, number of models to train
LR=0.0005
F=2  # the dropout ratio denominator
# the main script for learning the different dropout models

for V in `seq 1 $VAR`; do  # can be performed in parallel

    #1. first learn the original models
    python train_fcn.py --dataset cifar10 --dataroot $DATAROOT -oroot $OUTROOT --name var-$V --depth $DEPTH --width $WIDTH --learning_rate $LR --nepoch 200 
    MODEL=$OUTROOT/var-$V/checkpoint.pth

#2. once done, train all the subnetworks on the task
    for EL in `seq 0 $DEPTH`; do  # can be performed in parallel
        [ "$EL" -lt  2 ] && LRA=0.001 || LRA=0.003;
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

# plot the path
if [ "$F" -eq 2 ]; then 
    python path.py --M1 $OUTROOT/var-1/checkpoint.pth --M2 $OUTROOT/var-2/checkpoint.pth --nameA "A-f$F" --nameB "B-f$F"
    python plot_path.py $OUTROOT/path/ 
fi


