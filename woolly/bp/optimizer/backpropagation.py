import torch
from torch.optim import SGD, Adam


torch.manual_seed(1)


def get_sgd_optimizer(model, lr, momentum=0, weight_decay=0):
    return SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )


def get_adam_optimizer(model, lr, weight_decay=0):
    return Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
