import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn.inits import glorot
from layers import GraphAttentionLayer, SpGraphAttentionLayer, NMF_Nodes, AltGraphAttentionLayer
from torch_geometric.nn.models import InnerProductDecoder
from torch_geometric.utils.loop import remove_self_loops, add_self_loops
from torch_geometric.utils import negative_sampling
import math


# def initH(topic_s, topic_e, input_feature):
#     H_list = []
#     for i in range(topic_s, topic_e + 1):
#         H_list.append(torch.zeros(size=(input_feature, topic_e, 1)))  # H is transposed
#         H_list[i - topic_s][:, i:] = 0
#         # TODO ensure the positive H
#         H_list[i - topic_s] = torch.abs(H_list[i - topic_s])
#
#     H_list = torch.cat([H for H in H_list], 2).cuda()
#     return H_list


def recon_loss(z, pos_edge_index, neg_edge_index=None):
    r"""Given latent variables :obj:`z`, computes the binary cross
    entropy loss for positive edges :obj:`pos_edge_index` and negative
    sampled edges.
    Args:
        z (Tensor): The latent space :math:`\mathbf{Z}`.
        pos_edge_index (LongTensor): The positive edges to train against.
        neg_edge_index (LongTensor, optional): The negative edges to train
            against. If not given, uses negative sampling to calculate
            negative edges. (default: :obj:`None`)
    """
    EPS = 1e-15
    decoder = InnerProductDecoder()

    pos_loss = -torch.log(
        decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

    # Do not include self-loops in negative samples
    pos_edge_index, _ = remove_self_loops(pos_edge_index)
    pos_edge_index, _ = add_self_loops(pos_edge_index)
    if neg_edge_index is None:
        neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
    neg_loss = -torch.log(1 - decoder(z, neg_edge_index,
                                      sigmoid=True) + EPS).mean()

    return pos_loss + neg_loss


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nheads, use_nmf=False):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.use_nmf = use_nmf

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        n_out_att_infeat = nhid * nheads

        if use_nmf:
            # H_list = initH(topic_s=500, topic_e=510, input_feature=nhid * nheads)
            self.nmf = NMF_Nodes(input_feature=nhid *
                                 nheads, topic_s=20, topic_e=25)
            n_out_att_infeat = nhid * nheads * 2

        self.out_att = GraphAttentionLayer(
            n_out_att_infeat, nclass, dropout=dropout, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)

        if self.use_nmf:
            # X_nmf = x.detach().clone()
            # x, W_nmf, H_nmf = self.NMFs(X_nmf)  # detach here to cut off the BP to other layers!

            x, hyper = self.nmf(x)  # hypergraph
            # x = self.nmf(x)  # default
            adj = adj - hyper  # hypergraph
        else:
            X_nmf = None
            W_nmf = None
            H_nmf = None

        # x = F.elu(self.out_att(x, adj))
        # x = F.log_softmax(x, dim=1)

        return x


class GATConv(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.nfeat = nfeat
        self.nhid = nhid
        self.nclass = nclass
        self.nheads = nheads

        self.linear = nn.Linear(nfeat, nhid * nheads)  # W
        self.att_l = nn.Parameter(torch.Tensor(1, nheads, nclass))
        self.att_r = nn.Parameter(torch.Tensor(1, nheads, nclass))

        glorot(self.linear.weight)
        glorot(self.att_l.weight)
        glorot(self.att_r.weight)
        # for i, attention in enumerate(self.attentions):
        # self.add_module('attention_{}'.format(i), attention)

        # n_out_att_infeat = nhid * nheads

        # self.out_att = GraphAttentionLayer(n_out_att_infeat, nclass, dropout=dropout, concat=False)

    def glorot(tensor):
        if tensor is not None:
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        N = adj.shape[0]
        # change: no initial dropout
        x_l = x_r = self.linear(x).view(-1, self.nheads, self.nclass)
        alpha_l = (x_l * self.att_l).sum(dim=-1)  # n_node * n_head
        alpha_r = (x_r * self.att_r).sum(dim=-1)

        # dist matrix is the outer product of the alpha
        att_mat = torch.bmm(alpha_l.t().unsqueeze(
            2), alpha_r.t().unsqueeze(1)).permute(1, 2, 0)  # n_node(l) * n_node(r) * n_head

        zero_vec = -9e15 * torch.ones_like(att_mat)
        att_mat = torch.where(torch.stack(
            [adj for _ in range(self.nheads)]) > 0, att_mat, zero_vec)  # mask the inconnected nodes
        att_mat = F.softmax(att_mat, dim=1)
        att_mat = F.dropout(att_mat, self.dropout, training=self.training)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu()

        return x


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


class AltGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nheads):
        """Dense version of GAT."""
        super(AltGAT, self).__init__()
        self.dropout = dropout

        self.attention = AltGraphAttentionLayer(
            nfeat, nclass * nheads, dropout=dropout, concat=True)
        # self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, concat=True) for _ in
        #    range(nheads)]
        # for i, attention in enumerate(self.attentions):
        # self.add_module('attention_{}'.format(i), attention)

        n_out_att_infeat = nclass * nheads
        self.out_att = AltGraphAttentionLayer(
            n_out_att_infeat, nclass, dropout=dropout, concat=False)
        self.bias = Parameter(torch.Tensor(n_out_att_infeat))
        # glorot(self.attention.weight)

    def glorot(tensor):
        if tensor is not None:
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

    def zeros(tensor):
        if tensor is not None:
            tensor.data.fill_(0)

    def forward(self, x, adj):
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.attention(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.out_att(x, adj)

        return x


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
