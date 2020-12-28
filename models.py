import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer, NMF_Nodes


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nheads, use_nmf):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.use_nmf = use_nmf

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads * 2, nclass, dropout=dropout, concat=False)

        if use_nmf:
            self.nmf = NMF_Nodes(input_feature=nhid * nheads, topic_s=500, topic_e=510)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)

        if self.use_nmf:
            # X_nmf = x.detach().clone()
            # x, W_nmf, H_nmf = self.NMFs(X_nmf)  # detach here to cut off the BP to other layers!
            # x, hyper = self.nmf(X_nmf)  # hypergraph
            x = self.nmf(x)  # default
            # adj = adj - hyper  # hypergraph
        else:
            X_nmf = None
            W_nmf = None
            H_nmf = None

        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


# class MFGAT(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
#         """Dense version of GAT."""
#         super(MFGAT, self).__init__()
#         print('MF', nhid)
#         self.dropout = dropout

#         self.attentions = [MFGraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
#                            range(nheads)]
#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)

#         self.out_att = MFGraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

#         self.H = nn.Parameter(torch.zeros(size=(nhid, nclass)))
#         nn.init.xavier_uniform_(self.H.data, gain=1.414)

#     def forward(self, x, adj):
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
#         x = F.dropout(x, self.dropout, training=self.training)
#         # add multi-head attention
#         print('x',x.shape)
#         x = F.elu(self.out_att(x, adj))
#         print('out',x.shape)
#         self.H=nn.Parameter(x)
#         return F.log_softmax(x, dim=1)


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat,
                                                 nhid,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads,
                                             nclass,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)
