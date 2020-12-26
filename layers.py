import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class NMF_Nodes(nn.Module):
    def __init__(self, input_feature,
                 topic_s=5, topic_e=10):
        super(NMF_Nodes, self).__init__()
        self.H_list = []
        self.topic_s = topic_s
        self.topic_e = topic_e
        for i in range(self.topic_s, self.topic_e + 1):
            self.H_list.append(nn.Parameter(torch.zeros(
                size=(input_feature, self.topic_e, 1))))
            # nn.init.xavier_uniform_(self.H_list[i - self.topic_s], gain=1.414)
            self.H_list[i - self.topic_s][:, i:] = 0
            # TODO ensure the positive H
            self.H_list[i - self.topic_s] = torch.abs(self.H_list[i - self.topic_s])
            # print(self.H_list[i-self.topic_s])

        self.H_list = torch.cat([H for H in self.H_list], 2).cuda()

        self.W = nn.Parameter(torch.zeros(size=(input_feature, 10)))
        # nn.init.xavier_uniform_(self.W.data, gain=1.414)
        # self.linear = nn.Linear(input_feature * 2, input_feature)
        # self.linear_do = nn.Dropout(p=.5)

    def forward(self, x):
        # W_list = [(x @ self.H_list[..., i]).unsqueeze(2) for i in range(self.topic_e - self.topic_s + 1)]
        W_list = []
        for i in range(self.topic_e - self.topic_s + 1):
            W = torch.matmul(x, self.H_list[:, :, i])
            W = W.unsqueeze(2)
            # print('Ws',W)
            W_list.append(W)
        W_list = torch.cat([W for W in W_list], 2)
        # print('W',W_list.shape)
        N, D = int(x.shape[0]), int(x.shape[1])
        # print('x shape',x.shape)
        # TODO similarity or attention?
        t1 = x.repeat(N, 1)
        t2 = x.repeat(1, N).reshape(-1, D)
        # FIXME change this
        t = torch.cat((t1, t2), 1)  # 
        sim = torch.cosine_similarity(t1, t2)
        sim = sim.unsqueeze(0)
        t = sim.T * t
        t = t.reshape(N, N, -1)

        W_max = torch.argmax(W_list, dim=1)
        # print(W_max, W_max.shape)
        W1 = W_max.repeat(N, 1).reshape(-1, N, self.topic_e - self.topic_s + 1)
        W2 = W_max.repeat(1, N).reshape(-1, N, self.topic_e - self.topic_s + 1)
        # print(W1)
        # print(W2)
        comp = torch.eq(W1, W2)
        topic_count = torch.sum(comp, dim=2, keepdim=True)
        topic_count = torch.div(topic_count.float(),
                                self.topic_e - self.topic_s + 1)

        new_n = topic_count * t
        new_nodes = torch.mean(new_n, dim=0)
        # print('nodes',new_nodes.shape)
        # new_nodes = self.linear_do(self.linear(new_nodes))
        return new_nodes



class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
