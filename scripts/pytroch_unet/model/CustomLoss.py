import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self, loss='BCEWithLogitsLoss', beta=0, lteReconstructionLoss=None):
        super(CustomLoss, self).__init__()
        self.reconstructionloss=0
        self.avgcosloss =0
        self.beta = beta
        self.kldloss = 0
        self.Mloss = getattr(nn, loss)
        self.mloss = self.Mloss()

    def forward(self, output, groundtrue):
        out, mu, v = output[0], output[1], output[2]
        self.reconstructionloss = self.mloss(out, groundtrue)
        if (type(v) is not int):
            self.kldloss = -0.5 * torch.sum(1 + v - mu.pow(2) - v.exp(), dim=(-2,-1))
            self.kldloss = self.kldloss.mean()
        #if (self.reconstructionloss < 1.15):
        #    return self.reconstructionloss + (self.beta * self.kldloss) + (self.beta * (1-self.avgcosloss))
        return self.reconstructionloss + (self.beta * self.kldloss)
