import sys
from argparse import Namespace
from contextlib import suppress
from typing import List

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim import Adam #Lin++





class ContinualModel(nn.Module):
    """
    Continual learning model.
    """
    NAME: str
    COMPATIBILITY: List[str]

    def __init__(self, backbone: nn.Module, loss: nn.Module,
                 args: Namespace, transform: nn.Module) -> None:
        super(ContinualModel, self).__init__()

        self.net = backbone
        self.loss = loss
        self.args = args
        self.transform = transform
        self.opt = Adam(self.net.parameters(), lr=self.args.lr)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        return self.net(x)

    def meta_observe(self, *args, **kwargs):
        # if 'wandb' in sys.modules and not self.args.nowand:
        #     pl = persistent_locals(self.observe)
        #     ret = pl(*args, **kwargs)
        #     self.autolog_wandb(pl.locals)
        # else:
        ret = self.observe(*args, **kwargs)
        return ret

    # def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
    #             not_aug_inputs: torch.Tensor) -> float:
    def observe(self, inputs: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :param kwargs: some methods could require additional parameters
        :return: the value of the loss function
        """
        raise NotImplementedError
