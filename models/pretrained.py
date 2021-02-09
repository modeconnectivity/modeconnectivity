from torchvision import models, datasets, transforms
import torch.nn as nn
import utils

def set_parameter_requires_grad(model, grad):

    for param in model.parameters():
        param.requires_grad = grad

def initialize_model(model_name, pretrained, feature_extract=True, freeze=False, num_classes=10, use_pretrained=True, *args, **kwargs) -> (nn.Module, int):

    if model_name.find('vgg') != -1:
        if model_name == 'vgg-16':
            model_ft = models.vgg16(pretrained=pretrained)
        elif model_name == 'vgg-11':
            model_ft = models.vgg11(pretrained=pretrained)
        #model_ft.requires_grad(not feature_extract)
        #set_parameter_requires_grad(model_ft, not feature_extract)

        num_ftrs = model_ft.classifier[6].in_features  # change the last layer

        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        new_class_lst = [n for n in model_ft.classifier if not type(n) is nn.Dropout]  # remove dropout
        input_size = 224
        model_ft.classifier = nn.Sequential(*new_class_lst)

        set_parameter_requires_grad(model_ft, not freeze)

        if feature_extract:
            if kwargs:
                depth = kwargs.get('depth', 2)
                width = kwargs.get('width', 500)
                widths = depth*[width]
                sizes = [num_ftrs, *widths, num_classes]
                #mlp = utils.construct_mlp_net(sizes, fct_act=nn.LeakyReLU, kwargs_act={'negative_slope': lrelu, 'inplace': True})
                mlp = utils.construct_mlp_net(sizes, fct_act=nn.ReLU, args_act=[True])
                model_ft.classifier = mlp

            set_parameter_requires_grad(model_ft.features, False)
            set_parameter_requires_grad(model_ft.classifier, True)

        #model_ft.requires_grad(not feature_extract)

    elif model_name.find('resnet') != -1:

        pass

    return model_ft, input_size

