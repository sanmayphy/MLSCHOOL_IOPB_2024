import torch#, torchvision, torchmetrics
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

# -- the deepset layer -- #
class DeepSetLayer(nn.Module):
    def __init__(self, in_features:int, out_features:int,  normalization:str = '', pool:str = 'mean') -> None :
        """
        DeepSets single layer
        :param in_features: input's number of features
        :param out_features: output's number of features
        :param attention: Whether to use attention
        :param normalization: normalization method - 'fro' or 'batchnorm'
        
        """
        super(DeepSetLayer, self).__init__()

        self.Gamma = nn.Linear(in_features, out_features)
        self.Lambda = nn.Linear(in_features, out_features)

        self.normalization = normalization
        self.pool = pool
        
        if normalization == 'batchnorm':
            self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x : torch.Tensor) -> torch.Tensor :
        # x.shape = (B, N, C)
        if(self.pool == 'mean') : 
            x = self.Gamma(x) + self.Lambda(x - x.mean(dim=1, keepdim=True)) # -- the average is over the points -- #
        elif(self.pool == 'max') :
            x = self.Gamma(x) + self.Lambda(x - x.max(dim=1, keepdim=True)) # -- the max is over the points -- #

        # normalization
        if self.normalization == 'batchnorm':
            x = self.bn(x)
        

        return x
    
# --------------------------------------------- #
# -- the deepset model -- #
class DeepSet(nn.Module):
    def __init__(self, in_features:int, feats:list, n_class:int, normalization:str = '', pool:str = 'mean') ->None:
        """
        DeepSets implementation
        :param in_features: input's number of features
        :param feats: list of features for each deepsets layer
        """
        super(DeepSet, self).__init__()

        layers = []

        layers.append(DeepSetLayer(in_features = in_features, out_features = feats[0], normalization = normalization, pool = pool))
        for i in range(1, len(feats)):
            layers.append(nn.ReLU())
            layers.append(DeepSetLayer(in_features = feats[i-1], out_features = feats[i], normalization = normalization, pool = pool))

        layers.append(DeepSetLayer(in_features = feats[-1], out_features = n_class, normalization = normalization, pool = pool))
        #self.sequential = nn.Sequential(*layers)
        self.sequential = nn.ModuleList(layers)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        #return self.sequential(x)
        for i, layer in enumerate(self.sequential):
            x = layer(x)
        
        x = x.mean(dim=1) # -- average over the points -- #
        out = F.log_softmax(x, dim=-1)
        
        return out

    
    