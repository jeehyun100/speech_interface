#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn

class Dvector(nn.Module):    
    def __init__(self, n_spks, indim, outdim):
        super(Dvector, self).__init__()
        self.n_spks = n_spks
        self.linears = nn.Sequential(nn.Linear(indim, outdim),
                                     nn.LeakyReLU(negative_slope=0.2),        
                                     nn.Linear(outdim, outdim),
                                     nn.LeakyReLU(negative_slope=0.2), 
                                     nn.Linear(outdim, 128),
                                     nn.LeakyReLU(negative_slope=0.2),
                                     nn.Linear(128, 128),
                                     nn.LeakyReLU(negative_slope=0.2))

        self.clf = nn.Linear(128, self.n_spks)
        
    def forward(self, x, extract=False):
        # Normalize input features.
        x_mean = torch.mean(x, -1)
        x_var = torch.std(x, -1)
        x_var[x_var < 0.01] = 0.01
        x = (x - x_mean[:,:,None]) / x_var[:,:,None]

        x = self.linears(x)

        x = x.mean(dim=1)
        if extract:
            x = self.clf(x)
            
        return x