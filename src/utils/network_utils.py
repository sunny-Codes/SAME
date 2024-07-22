from torch import nn
import torch, copy
import numpy as np


def mlp(
    sizes,
    activation,
    output_activation=nn.Identity(),
    use_batchnorm=False,
    dropout_p=None,
):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1])]
        if use_batchnorm:
            layers += nn.BatchNorm1d(sizes[j + 1])
        layers += [act]
        if dropout_p is not None:
            layers += [nn.Dropout(p=dropout_p)]
        # if use_batchnorm:
        #     layers += [nn.Linear(sizes[j], sizes[j+1]), nn.BatchNorm1d(sizes[j+1]), act]
        # else: layers += [nn.Linear(sizes[j], sizes[j+1]), act]
    return nn.Sequential(*layers)


def get_scheduler(optimizer, scheduler_dict=None, epoch_num=None):
    if scheduler_dict is None:
        return None

    scheduler_type = scheduler_dict["type"]
    scheduler_dict.pop("type")

    if scheduler_type == "exponential":

        def lambda_rule(epoch):
            lr_l = max(scheduler_dict["min"], scheduler_dict["gamma"] ** epoch)
            return lr_l

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        # return torch.optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_dict)
    if scheduler_type == "linear":
        assert epoch_num != None, "linear schedule but epoch is None"

        def lambda_rule(epoch):
            lr_l = max(
                scheduler_dict["min"], 1 - epoch * scheduler_dict["slope"]
            )  # 1/float(epoch_num+1)
            return lr_l

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    if scheduler_type == "Step_LR":
        print("Step_LR scheduler set")
        return torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_dict)
        # return torch.optim.lr_scheduler.StepLR(optimizer, 600, 0.5)
    # if scheduler_type == 'Plateau':
    #     print('Plateau_LR shceduler set')
    #     return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_dict)
    # return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5, verbose=True)


def count_param(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    return num_params
