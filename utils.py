import numpy as np
import scipy.sparse as sp
import torch
from pandas import DataFrame
from surprise import SVD, Reader, Dataset
from surprise.model_selection import cross_validate


def load_raw(path="./data/cora/", dataset="cora"):
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    return features, labels, edges_unordered


def svd_train():
    idx_features_labels, _, _ = load_raw()
    rd = Reader(rating_scale=(0, 1))
    idx_features_labels = DataFrame(idx_features_labels)
    data = Dataset.load_from_df(df=idx_features_labels, reader=rd)
    alg = SVD()
    cross_validate(alg, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="./data/cora/", dataset="cora", sample=False):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    if sample:
        idx_features_labels, edges_unordered = dataset_sample(path, dataset, sample_factor=5, verbose=True)
    else:
        idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
        edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)

    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
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
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


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
    idx_features_labels = np.genfromtxt("{}{}.content".format(pth, dataset), dtype=np.dtype(str))

    edges_unordered = np.genfromtxt("{}{}.cites".format(pth, dataset), dtype=np.int32)
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
        if len(seen_new) == 0: break
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
    idx_sample = [np.where(idx_features_labels[:, 0] == str(x))[0].item() for x in seen]
    idx_features_sample = idx_features_labels[idx_sample]
    return idx_features_sample, edges_sample

    # idx_features_labels = idx_features_labels[np.random.choice(cnt, int(.1 * cnt)), :]
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])


if __name__ == '__main__':
    svd_train()
