import numpy as np
import scipy.sparse as sp
import torch
from pandas import DataFrame
# from surprise import SVD, Reader, Dataset
# from surprise.model_selection import cross_validate
from sklearn.decomposition import non_negative_factorization
from torch_geometric import datasets
from torch_geometric.utils import to_undirected
from sklearn.metrics import roc_auc_score, average_precision_score
import math
from torch_geometric.nn.models import InnerProductDecoder



def load_raw(path="./data/cora/", dataset="cora"):
    idx_features_labels = np.genfromtxt(
        "{}{}.content".format(path, dataset), dtype=np.dtype(str))
    edges_unordered = np.genfromtxt(
        "{}{}.cites".format(path, dataset), dtype=np.int32)
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    return features, labels, edges_unordered


def nmf_train(sp, n_topic):
    sp_df = DataFrame(sp)
    model = NMF(n_components=n_topic, init="random",
                random_state=0, verbose=True)
    W = model.fit_transform(sp_df)
    H = model.components_
    # W, H = torch.from_numpy(W), torch.from_numpy(H)
    return W, H


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[
                       i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(
        list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="./data/cora/", dataset="cora", sample=False):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    if sample:
        idx_features_labels, edges_unordered = dataset_sample(
            path, dataset, sample_factor=5, verbose=True)
    else:
        idx_features_labels = np.genfromtxt(
            "{}{}.content".format(path, dataset), dtype=np.dtype(str))
        edges_unordered = np.genfromtxt(
            "{}{}.cites".format(path, dataset), dtype=np.int32)
        features = sp.csr_matrix(
            idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = encode_onehot(idx_features_labels[:, -1])

    features = features.todense()

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    edges = torch.from_numpy(edges.T).long()

    return adj, features, labels, idx_train, idx_val, idx_test, edges


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).T.dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def dataset_sample(pth="./data/cora/", dataset="cora", sample_factor=3, verbose=False):
    idx_features_labels = np.genfromtxt(
        "{}{}.content".format(pth, dataset), dtype=np.dtype(str))

    edges_unordered = np.genfromtxt(
        "{}{}.cites".format(pth, dataset), dtype=np.int32)
    edges_sample = []
    # get paper id
    source_nodes = np.unique(edges_unordered[:, 0])
    idx = np.array(source_nodes, dtype=np.int32)

    # randomly choose paper from idx set
    sample_idx = idx[np.random.choice(len(idx), sample_factor)]
    print("randomly chosen samples: {}".format(sample_idx))

    # traverse graph to obtain connected individuals
    seen = sample_idx
    seen_last = seen
    i = 0

    while True:
        # obtain source nodes
        seen_new = np.array([])
        for v in seen_last:
            conn = edges_unordered[np.where(edges_unordered[:, 0] == v)[0], 1]
            for c in conn:
                if c in seen:
                    continue
                seen = np.append(seen, c)
                seen_new = np.append(seen_new, c)
                edges_sample.append([v, c])
        if verbose:
            print("step {}, add {} nodes".format(i, len(seen_new)))
        i += 1
        if len(seen_new) == 0:
            break
        seen_last = seen_new
    edges_sample = np.array(edges_sample, dtype=np.int32)
    if verbose:
        print("sampled {} nodes, are {}".format(len(seen), seen))
        print(edges_sample)
        # for v in seen:
        #     conn = edges_unordered[np.where(edges_unordered[:, 0] == v)[0], 1]
        #     print("{}: {}".format(v, conn))
        #     for c in conn:
        #         print("{} is at position {}".format(c, np.where(seen == v)[0]))
    seen = seen.astype(int)
    idx_sample = [np.where(idx_features_labels[:, 0] == str(x))[
                      0].item() for x in seen]
    idx_features_sample = idx_features_labels[idx_sample]
    return idx_features_sample, edges_sample

    # idx_features_labels = idx_features_labels[np.random.choice(cnt, int(.1 * cnt)), :]
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])


def load_pubmed():
    ple = datasets.Planetoid(root="./datasets/", name="PubMed")
    data = ple[0]
    return data.x, data.y, data.train_mask, data.val_mask, data.test_mask


# def nmf_optim_ours(X, Ws, Hs, alpha=0., l1_ratio=0.):
#     epochs = 1000
#     Ws_new = torch.empty_like(Ws)
#     Hs_new = torch.empty_like(Hs)
#     for i in range(Ws.shape[-1]):
#         W = Ws[..., i]
#         H = Hs[..., i]
#         H = H.T
#         for epoch in range(epochs):
#             H_1 = torch.mm(W.T, X)
#             H_2 = torch.mm(torch.mm(W.T, W), H)
#             H_tr = H_1 / H_2
#             H_new = torch.mm(H, H_tr.T)
# 
#             W_1 = torch.mm(X, H.T)
#             W_2 = torch.mm(torch.mm(W, H), H.T)
#             W_tr = W_1 / W_2
#             W_new = torch.mm(W, W_tr.T)
# 
#             div_H = H_new - H
#             div_W = W_new - W
#             print("iter {}, div_W = {}, div_H = {}".format(epoch, div_W, div_H))
# 
#             H = H_new
#             W = W_new
#         Ws_new[..., i] = W
#         Hs_new[..., i] = H
#     return Ws_new, Hs_new


def nmf_optim(X, Ws, Hs, alpha=0., l1_ratio=0.):
    epochs = 1000
    n_sample = Ws.shape[-1]
    Ws = Ws.detach().data.cpu().numpy()
    Hs = Hs.detach().data.cpu().numpy()
    X = X.detach().data.cpu().numpy()
    Ws_new = np.empty_like(Ws)
    Hs_new = np.empty_like(Hs)

    for i in range(n_sample):
        H = Hs[..., i]
        W = Ws[..., i]
        H = H.T
        W_upd, H_upd, n_iter = non_negative_factorization(X, W=W, H=H, n_components=H.shape[0])
        print("n_iter = {}".format(n_iter))
        Ws_new[..., -1] = W_upd
        Hs_new[..., -1] = H_upd.T

    return Ws_new, Hs_new


def adj2edgeidx(adj):
    adj_u = np.triu(adj)
    existed_links = np.where(adj_u != 0)
    return torch.from_numpy(existed_links.T).cuda()


def train_test_split_edges(input_edge_index, num_nodes, val_ratio=0.05, test_ratio=0.1):
    r"""Splits the edges of a :obj:`torch_geometric.data.Data` object
    into positive and negative train/val/test edges, and adds attributes of
    `train_pos_edge_index`, `train_neg_adj_mask`, `val_pos_edge_index`,
    `val_neg_edge_index`, `test_pos_edge_index`, and `test_neg_edge_index`
    to :attr:`data`.

    Args:
        input_edge_index: the original pyg-styled edge index array (2 * n_edge)
        num_nodes: the number of nodes of the graph
        val_ratio (float, optional): The ratio of positive validation
            edges. (default: :obj:`0.05`)
        test_ratio (float, optional): The ratio of positive test
            edges. (default: :obj:`0.1`)

    :rtype: :class:`torch_geometric.data.Data`
    """

    row, col = input_edge_index
    edge_index = None

    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    # Positive edges.
    # TODO: correctly return values
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]

    r, c = row[:n_v], col[:n_v]
    val_pos_edge_index = torch.stack([r, c], dim=0)
    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    test_pos_edge_index = torch.stack([r, c], dim=0)

    r, c = row[n_v + n_t:], col[n_v + n_t:]
    train_pos_edge_index = torch.stack([r, c], dim=0)
    train_pos_edge_index = to_undirected(train_pos_edge_index)

    # Negative edges.
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0

    neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
    perm = torch.randperm(neg_row.size(0))[:n_v + n_t]
    neg_row, neg_col = neg_row[perm], neg_col[perm]

    neg_adj_mask[neg_row, neg_col] = 0
    train_neg_adj_mask = neg_adj_mask

    row, col = neg_row[:n_v], neg_col[:n_v]
    val_neg_edge_index = torch.stack([row, col], dim=0)

    row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
    test_neg_edge_index = torch.stack([row, col], dim=0)

    # return train_pos_edge_index, val_pos_edge_index, val_neg_edge_index, test_pos_edge_index, test_neg_edge_index
    return train_pos_edge_index, val_pos_edge_index, val_neg_edge_index


def validate(z, pos_pred, neg_pred, pos_edge_index, neg_edge_index):
    r"""Given latent variables :obj:`z`, positive edges
    :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
    computes area under the ROC curve (AUC) and average precision (AP)
    scores.

    Args:
        z (Tensor): The latent space :math:`\mathbf{Z}`.
        pos_edge_index (LongTensor): The positive edges to evaluate
            against.
        neg_edge_index (LongTensor): The negative edges to evaluate
            against.
    """
    decoder = InnerProductDecoder()

    pos_y = z.new_ones(pos_edge_index.size(1))
    neg_y = z.new_zeros(neg_edge_index.size(1))
    y = torch.cat([pos_y, neg_y], dim=0)

    pos_pred = decoder(z, pos_edge_index, sigmoid=True)
    neg_pred = decoder(z, neg_edge_index, sigmoid=True)
    pred = torch.cat([pos_pred, neg_pred], dim=0)

    y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

    return roc_auc_score(y, pred), average_precision_score(y, pred)


def edgeidx2adj(edges, n_node):
    edges = edges.T
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(n_node, n_node),
                        dtype=np.float32)
    return torch.LongTensor(adj.todense())


if __name__ == '__main__':
    adj, features, labels, idx_train, idx_val, idx_test, edge_index = load_data()
    n_nodes, n_feature = features.shape
    train_pos_edge_index = train_test_split_edges(edge_index, n_nodes)

