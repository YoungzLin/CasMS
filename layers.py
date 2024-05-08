#encoding:utf-8
import torch
from torch_geometric.nn import GCNConv
from torch.autograd import Variable

from TCN import TemporalConvNet

class my_GCN(torch.nn.Module):
    def __init__(self,in_channels, out_channels,filters_1, filters_2, dropout, bais=True):
        super(my_GCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filters_1 = filters_1
        self.filters_2 = filters_2
        self.dropout = dropout
        self.setup_layers()

    def setup_layers(self):
        self.convolution_1 = GCNConv(self.in_channels, self.filters_1)
        self.convolution_2 = GCNConv(self.filters_1, self.filters_2)
        self.convolution_3 = GCNConv(self.filters_2, self.out_channels)

    def forward(self, edge_indices, features):
        features = self.convolution_1(features, edge_indices)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.dropout,
                                               training=self.training)
        features = self.convolution_2(features, edge_indices)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.dropout,
                                               training=self.training)

        features = self.convolution_3(features, edge_indices)
        return features

class dynamic_routing(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(dynamic_routing,self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W = torch.nn.Parameter(torch.randn(1,in_dim,out_dim))

    def forward(self,x):
        num_nodes = x.size(1) 
        batch_size = x.size(0)
        W = torch.cat([self.W] * batch_size, dim=0)
        representation = torch.matmul(x, W)
        r_sum = torch.sum(representation, dim=-1, keepdim=False)
        b = torch.zeros([batch_size, num_nodes])
        b = Variable(b)
        one = torch.ones_like(r_sum)
        zero = torch.zeros_like(r_sum)
        label = torch.clone(r_sum)
        label = torch.where(label == 0, one, zero)
        b.data.masked_fill_(label.bool(), -float('inf'))
        num_iterations = 3
        for i in range(num_iterations):
            c = torch.nn.functional.softmax(b, dim=-1)
            weight_coeff = c.unsqueeze(dim=1)
            representation_global = torch.matmul(weight_coeff, representation)
            representation_global_all = torch.cat([representation_global] * num_nodes, dim=1)
            representation_similarity = torch.nn.functional.cosine_similarity(representation, representation_global_all, dim=-1)
            representation_similarity.data.masked_fill_(label.bool(), -float('inf'))
            b = representation_similarity
        return representation_global.squeeze(dim=1)

class my_TCN(torch.nn.Module):
    def __init__(self, tcn_inputsize, tcn_hiddensize, tcn_layers, tcn_dropout):
        super(my_TCN, self).__init__()
        self.tcn_inputsize = tcn_inputsize
        self.tcn_hiddensize = tcn_hiddensize
        self.tcn_layers = tcn_layers
        self.tcn_dropout = tcn_dropout
        self.setup_layers()

    def setup_layers(self):
        self.tcn = TemporalConvNet(
            num_inputs = self.tcn_inputsize,
            num_channels = [self.tcn_hiddensize] * self.tcn_layers,
            kernel_size=3, 
            dropout=self.tcn_dropout
        )

    def forward(self, input):
        out = self.tcn(input.transpose(1, 2))
        return out[:, :, -1]

class dens_Net(torch.nn.Module):
    def __init__(self, dens_inputsize, dens_hiddensize, dens_dropout, dens_outputsize):
        super(dens_Net, self).__init__()
        self.inputsize = dens_inputsize
        self.dens_hiddensize = dens_hiddensize
        self.dens_dropout = dens_dropout
        self.outputsize = dens_outputsize
        self.setup_layers()

    def setup_layers(self):
        self.dens_net = torch.nn.Sequential(
            torch.nn.Linear(self.inputsize, self.dens_hiddensize),
            torch.nn.Dropout(p=self.dens_dropout),
            torch.nn.Linear(self.dens_hiddensize, self.dens_hiddensize),
            torch.nn.Dropout(p=self.dens_dropout),
            torch.nn.Linear(self.dens_hiddensize, self.outputsize)
        )

    def forward(self, x):
        return self.dens_net(x)
