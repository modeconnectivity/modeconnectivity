# When are Solutions Connected in Deep Networks?

This repository is the official implementation of [When are Solutions Connected in Deep Networks](https://arxiv.org/abs/2102.09671). 

## Requirements

 * Python 3.7
 * PyTorch 1.7
 * Pandas

The detailed package requirement is given in ```requirements.txt``` 
To install requirements:

```setup
pip install -r requirements.txt
```

The datasets (MNIST, CIFAR-10) will be downloaded by PyTorch, default dataroot: ```data/```

## Workflow


### Layer Inspection


The scripts ```main_{mnist_fcn,cifar10_fcn,cifar10_vgg}.sh``` give the workflow
to reproduce the figures appearing in the article for the respective models and
datasets, including the average over several SGD solutions.

Simply run them to train and evaluate the models. 

Alternatively, to train the original model, (O), on e.g. CIFAR-10, run
```bash
python train_fcn.py --dataset cifar10 -oroot results/cifar10 --depth 5 --width 500 
```

Then, to conduct experiment (A):
```bash
for $EL in `seq 0 5`; do   # the different layers at which to prune the neurons
python exp_a_fcn.py --model results/cifar10/checkpoint.pth  --entry_layer $EL &
done
wait
python merge_a_fcn.py results/cifar10/
```

Experiment (B):
```bash
python exp_b.py --model results/cifar10/checkpoint.pth 
```

Plotting of the two:
```bash
python plot_meta.py results/cifar10/
```

### Path

Assuming having trained two models and run experiments A and B for both of them, a path connecting two solutions with bounded train loss can be constructed with
```bash
python path.py --M1 model1/checkpoint.pth --M2 model2/checkpoint.pth  --nameA A --nameB B 
```

### Varying Width Experiment


To peform the varying width experiment, first models with different widths. Then, ``merge'' them using 
```bash 
    python merge_max_widths.py path/to/model/root1 path/to/model/root2 ...
```
and plot them using 
```bash
python plot_widths.py path/to/model/merge/
```



## Contributing

MIT License


