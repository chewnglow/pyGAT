from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch_geometric.nn import GCNConv, GAE, VGAE, GATConv

from utils import load_data, accuracy, train_test_split_edges, edgeidx2adj
from models import GAT, SpGAT, recon_loss, AltGAT

from sklearn.decomposition import NMF

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true',
                    default=False, help='Disables CUDA training.')
parser.add_argument('--parallel', action='store_true',
                    default=False, help='Enables multi-GPU computing.')
parser.add_argument('--fastmode', action='store_true',
                    default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true',
                    default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=114514, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-2,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8,
                    help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=1,
                    help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6,
                    help='Dropout rate (1 - keep probability).')
# parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the.')
parser.add_argument('--patience', type=int, default=200, help='Patience')
parser.add_argument('--use_nmf', dest="use_nmf", default=False,
                    action="store_true", help='Whether to use NMF')
# parser.add_argument('--n-topic', dest="n_topic", default=514, help='the topic count of NMF')
parser.add_argument('--link', dest="link", default=True, action="store_true",
                    help='Whether to do the link prediction')

args = parser.parse_args()
use_nmf = args.use_nmf
print("use_nmf = {}".format(use_nmf))
args.cuda = not args.no_cuda and torch.cuda.is_available()

# random.seed(args.seed)
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.set_device(2)
    # torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test, edge_index = load_data()
n_nodes, n_feature = features.shape

if args.link:
    train_pos_edge_index, val_pos_edge_index, val_neg_edge_index = train_test_split_edges(
        edge_index, n_nodes)
    train_pos_adj = edgeidx2adj(train_pos_edge_index, n_nodes)
    val_pos_adj = edgeidx2adj(val_pos_edge_index, n_nodes)
    val_neg_adj = edgeidx2adj(val_neg_edge_index, n_nodes)


# class GCNEncoder(torch.nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(GCNEncoder, self).__init__()
#         self.conv1 = GATConv(in_channels, 2 * out_channels)
#         self.conv2 = GATConv(2 * out_channels, out_channels)

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index).relu()
#         return self.conv2(x, edge_index)


# Model and optimizer
if args.sparse:
    model = SpGAT(nfeat=n_feature,
                  nhid=args.hidden,
                  nclass=int(labels.max()) + 1,
                  dropout=args.dropout,
                  nheads=args.nb_heads)
else:
    if not args.link:  # node prediction
        model = GAT(nfeat=n_feature,
                    nhid=args.hidden,
                    nclass=int(labels.max()) + 1,
                    dropout=args.dropout,
                    nheads=args.nb_heads,
                    use_nmf=use_nmf)
    else:  # edge prediction
        model = AltGAT(nfeat=n_feature,
                       nhid=32,
                       nclass=16,
                       dropout=args.dropout,
                       nheads=1)

# model = GAE(GCNEncoder(n_feature, 16))

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay)
# optim_nmf = optim.Adam(model.parameters(),
#                        lr=args.lr_nmf,
#                        weight_decay=args.weight_decay_nmf)

if args.cuda:
    if args.parallel:
        model = nn.DataParallel(model)
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    if args.link:
        edge_index = edge_index.cuda()
        train_pos_edge_index = train_pos_edge_index.cuda()
        train_pos_adj = train_pos_adj.cuda()

features, adj, labels = Variable(features), Variable(adj), Variable(labels)


def train(epoch):
    torch.autograd.set_detect_anomaly(True)
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, train_pos_adj)
    # output = model.encode(features, train_pos_edge_index)

    # loss & optim
    if args.link:
        loss_train = recon_loss(output, train_pos_edge_index)

        # TODO: add variational loss
    else:
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    # if use_nmf:
    #     Ws_new, Hs_new = nmf_optim(X, Ws, Hs)
    #     state_dict = model.state_dict()
    #     state_dict['W_list.weight'] = Ws_new
    #     state_dict['H_list.weight'] = Hs_new
    #     model.load_state_dict(state_dict)

    #     loss_train_nmf = nmf_loss(X, Ws, Hs)
    #     loss_train_nmf.backward()
    #     optim_nmf.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        with torch.no_grad():
            output = model(features, adj)
            # output = model.encode(features, train_pos_edge_index)
    # TODO: autoencoder validation
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          # 'acc_train: {:.4f}'.format(acc_train.data.item()),
          # 'loss_val: {:.4f}'.format(loss_val.data.item()),
          # 'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item()


def compute_test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data[0]),
          "accuracy= {:.4f}".format(acc_test.data[0]))


# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
for epoch in range(args.epochs):
    loss_values.append(train(epoch))

    # save model
    torch.save(model.state_dict(), '{}.pkl'.format(epoch))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

# Testing
compute_test()
