import torch
import numpy as np
import scipy.io as sio
import scipy.sparse as ss
from sklearn.preprocessing import normalize
import scipy.sparse as sp
import random
import torch.nn.functional as F
from tqdm import trange
from torch_geometric.utils import k_hop_subgraph


def load_data(args, device):
    if args.fix_seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    data = sio.loadmat(args.path + args.dataset + '.mat')

    if args.normal:
        features = normalize(data['X'])
    else:
        features = data['X']
    if ss.isspmatrix(features):
        features = features.todense()
    features = torch.from_numpy(features).float().to(device)

    feature_distance = pairwise_distance(features.float().to(device))

    feature_distance = F.normalize(feature_distance)
    edge_rate = feature_distance.max() * args.edge_rate
    # print("max_dis:{}".format(feature_distance.max()))
    kthvalue = torch.kthvalue(
        feature_distance.view(feature_distance.shape[0] * feature_distance.shape[1], 1).T,
        int(feature_distance.shape[0] * feature_distance.shape[1] * edge_rate))[0]
    mask = (feature_distance > kthvalue).detach().float()
    feature_dis = (feature_distance * mask)

    adj_hat_list = []

    labels = data['Y'].flatten()
    labels = labels - min(set(labels))

    class_idx, class_num = count_idx(labels)
    idx_labeled, idx_unlabeled, idx_val = divide(class_idx, class_num, args.ratio)
    labels = torch.from_numpy(labels).long()

    adjs = data['adj'][0]
    for i in range(adjs.shape[0]):
        adj = ss.coo_matrix(adjs[i])
        adj_hat = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj_hat = adj_hat + sp.eye(adj.shape[0])
        rowsum = np.array(adj_hat.sum(1))
        degree_mat_inv_sqrt = ss.diags(np.power(np.maximum(rowsum, 1e-10), -0.5).flatten())
        adj_hat = adj_hat.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
        adj_hat = torch.from_numpy(adj_hat.todense()).float()

        adj_hat = construct_sparse_float_tensor(adj_hat).to(device)
        adj_hat_list.append(adj_hat)

        del adj_hat
        torch.cuda.empty_cache()
    torch.cuda.empty_cache()

    return features, feature_dis, adj_hat_list, labels, idx_labeled, idx_unlabeled, idx_val


def construct_sparse_float_tensor(np_matrix):
    sp_matrix = ss.csc_matrix(np_matrix)
    three_tuple = sparse_to_tuple(sp_matrix)
    sparse_tensor = torch.sparse.FloatTensor(torch.LongTensor(three_tuple[0].T),
                                             torch.FloatTensor(three_tuple[1]),
                                             torch.Size(three_tuple[2]))
    return sparse_tensor


def sparse_to_tuple(sparse_mx):
    if not ss.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def count_idx(labels):
    dic_idx = {}
    dic_count = {}
    for index, c in enumerate(labels):
        if c not in dic_idx.keys():
            dic_idx[c] = []
            dic_count[c] = 0
        dic_idx[c].append(index)
        dic_count[c] += 1
    return dic_idx, dic_count


def divide(dic1, dic2, ratio):
    keys = list(dic1.keys())
    idx_train = []
    idx_test = []
    idx_val = []
    for i in keys:
        random.shuffle(dic1[i])
        num1 = int(dic2[i] * ratio)
        num2 = int(dic2[i] * 0.1)
        idx_train.extend(dic1[i][0:num1])
        idx_val.extend(dic1[i][num1:num1 + num2])
        idx_test.extend(dic1[i][num1 + num2:])
    return idx_train, idx_test, idx_val


def pairwise_distance(x, y=None):
    x = x.unsqueeze(0).permute(0, 2, 1)
    if y is None:
        y = x
    y = y.permute(0, 2, 1)
    A = -2 * torch.bmm(y, x)
    A += torch.sum(y ** 2, dim=2, keepdim=True)
    A += torch.sum(x ** 2, dim=1, keepdim=True)
    return A.squeeze()



