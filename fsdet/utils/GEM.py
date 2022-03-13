import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from fsdet.utils.events import get_event_storage

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1, device="cuda:0") * p)
        self.eps = eps
        print("Using GeM")

    def forward(self, x):

        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # print(p)
        storage = get_event_storage()
        storage.put_scalar("GeM/p", p)
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

    def __repr__(self):
        return self.__class__.__name__ + \
               '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
               ', ' + 'eps=' + str(self.eps) + ')'
