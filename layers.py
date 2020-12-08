import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def similarity(x1,x2):
    sim=torch.cosine_similarity(x1,x2,dim=0)
    x=sim*torch.cat((x1,x2),0)
    return x

class NMF_Nodes(nn.Module):
    def __init__(self,input_feature,
                 topic_s=3,topic_e=10):
        super(NMF_Nodes, self).__init__()
        self.H_list=[]
        self.topic_s=topic_s
        self.topic_e=topic_e
        for i in range(self.topic_s,self.topic_e+1):
            self.H_list.append(nn.Parameter(torch.zeros(size=(input_feature,self.topic_e,1))))
            nn.init.xavier_uniform_(self.H_list[i-self.topic_s],gain=1.414)
            self.H_list[i-self.topic_s][:,i:]=0
            # TODO ensure the positive H
            self.H_list[i-self.topic_s]=torch.abs(self.H_list[i-self.topic_s])
            # print(self.H_list[i-self.topic_s])

        self.H_list=torch.cat([H for H in self.H_list],2).cuda()

        self.W = nn.Parameter(torch.zeros(size=(input_feature,10)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

    def forward(self, x):
        W_list=[]
        for i in range(self.topic_e-self.topic_s):
            W=torch.matmul(x,self.H_list[:,:,i])
            W=W.unsqueeze(2)
            # print('Ws',W)
            W_list.append(W)
        W_list=torch.cat([W for W in W_list],2)
        # print('W',W_list.shape)
        nodes_num=int(x.shape[0])
        #TODO similarity or attention?
        new_nodes=[]
        for i in range(nodes_num):
            anchor_node=similarity(x[i],x[i])
            for j in range(nodes_num):
                if j==i:continue
                anchor_node_j=similarity(x[i],x[j])
                counts=0
                for t in range(self.topic_e-self.topic_s):
                    max_si=torch.argmax(W_list[i,:,t])
                    max_s2=torch.argmax(W_list[j,:,t])
                    if max_si==max_s2:
                        counts+=1
                anchor_node = anchor_node + counts/(self.topic_e-self.topic_s)*anchor_node_j
            anchor_node = anchor_node/nodes_num
            new_nodes.append(anchor_node.unsqueeze(0))
        new_nodes=torch.cat([n for n in new_nodes],0)
        # print('nodes',new_nodes.shape)
        return new_nodes

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True,NMF=False):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.NMFs=NMF_Nodes(input_feature=in_features)

    def forward(self, input, adj):
        # adding weight for input
        # adj means adjency matrix
        print('att h0', input.shape)


        h = torch.mm(input, self.W)
        N = h.size()[0]
        print('att h',h.shape)

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1)

        # print(a_input.shape)
        a_input = a_input.view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # print('e',e.shape)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # h is the self infor, this step means update
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class MFGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(MFGraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.D=nn.Parameter(torch.zeros(size=(out_features,out_features)))
        nn.init.xavier_uniform_(self.D.data,gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.NMFs = NMF_Nodes(input_feature=in_features)

    def forward(self, input, adj):
        # adding weight for input
        # adj means adjency matrix
        print('h0', input.shape)
        self.NMFs(input)
        h = torch.mm(input, self.W)
        N = h.size()[0]
        print('h',h.shape)

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1)

        print('a',a_input.shape)
        a_input = a_input.view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # print('e',e.shape)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # h is the self infor, this step means update
        h_prime = torch.matmul(attention, h)
        print('h1',h_prime.shape)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
