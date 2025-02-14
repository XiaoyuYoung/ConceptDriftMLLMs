import math
import torch
from os import path
import torch.nn as nn
import torch.nn.functional as F


class tSP_Embedding(nn.Linear):

    def __init__(self, in_features, out_features, bias=False):
        super(tSP_Embedding, self).__init__(in_features, out_features, bias)
        self.s_ = torch.nn.Parameter(torch.zeros(1))
        self.kappa = 16

    def forward(self, input):

        cosine = F.linear(
            F.normalize(input, p=2, dim=1),
            F.normalize(self.weight, p=2, dim=1),
            None
        )

        logit =  2 / (1 + self.kappa * (1-cosine))
        logit = logit * F.softplus(self.s_).add(1.)
        return logit
    
    def predict(self):
        return  F.linear(
            F.normalize(input, p=2, dim=1),
            F.normalize(self.weight, p=2, dim=1),
            None
        )


class Cosine_Embedding(nn.Linear):

    def __init__(self, in_features, out_features, bias=False):
        super(Cosine_Embedding, self).__init__(in_features, out_features, bias)

    def forward(self, input):
        logit = F.linear(
            F.normalize(input, p=2, dim=1),
            F.normalize(self.weight, p=2, dim=1),
            None
        )  

        return logit