# When are Solutions Connected in Deep Networks?

This repository is the official implementation of [When are Solutions Connected in Deep Networks](https://arxiv.org/abs/2102.09671). 

<!-->>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials <-->
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



The scripts ```main_{mnist_fcn,cifar10_fcn,cifar10_vgg}.sh``` give the workflow
to reproduce the figures appearing in the article for the respective models and
datasets.

Simply run them to train and evaluate the models. 


## Pre-trained Models TODO!

You can download pretrained models here (TODO!):

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 




## Results TODO!

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

MIT License


