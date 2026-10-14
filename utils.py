import pickle as pkl
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
# import networkx as nx
# from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import random
import math
from sklearn import metrics
import os
import torch.utils.data as data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import scipy.io as scio
from sklearn import metrics
# from munkres import Munkres
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
# from preprocessing import create_trainvaltest_split, \
#     sparse_to_tuple, preprocess_user_item_features, globally_normalize_bipartite_adjacency, \
#     load_data_monti, load_official_trainvaltest_split, normalize_features


# ------------------------------------
# Some functions borrowed from:
# https://github.com/tkipf/pygcn and
# https://github.com/tkipf/gae
# ------------------------------------


def get_data(X, labels, index):
#     miniH = H.copy().toarray()[index]
#     colsum = np.sum(miniH, 0)
#     idx_zero = np.where(colsum == 0)[0]
#     miniH = sp.csr_matrix(np.delete(miniH, idx_zero, axis=1))
    
    return X[index], labels[index]


class MyDataset(Dataset):
    def __init__(self, x, y):
        self.len = x.shape[0]
        self.x = x
        self.y = y
#         self.label = label
        
    def __getitem__(self, index):
        return self.x[index], self.y[index], index
    
    def __len__(self):
        return self.len
    


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    

def seed_everything(seed=616):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def get_edge_features_sparse(h, X):

    node_degree = np.array(h.sum(1)).reshape(-1)
    Y = sp.lil_matrix((h.shape[1], X.shape[1]))

    for i in range(Y.shape[0]):
        
        non_zero = h[:, i].nonzero()[0]        
        deg = node_degree[non_zero]
        weight = deg / np.sum(deg)
        
        fea = []
        for j in range(len(non_zero)):
            fea.append(weight[j] * X[non_zero[j], :])
        fea = np.array(fea)
#         Y[i, :] = sp.lil_matrix(fea.sum(0).reshape(-1))
        Y[i, :] = sp.lil_matrix(fea.sum(0))
    return Y

"""
def load_data_sparse(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    h = adj.copy()
    
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    edge_features = get_edge_features_sparse(h, features)
    
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)
    
    #if dataset_str != 'citeseer':
    features = normalize_features(features)
    edge_features = normalize_features(edge_features)

    return h, features, edge_features, labels, idx_train, idx_val, idx_test

def load_data_sparse_trials(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    h = adj.copy()
    
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    edge_features = get_edge_features_sparse(h, features)
    
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    
    #if dataset_str != 'citeseer':
    features = normalize_features(features)
    edge_features = normalize_features(edge_features)
    
    idx_test = test_idx_range
    seed_list = np.arange(2000)
    idx_train_list = []
    idx_val_list = []
#     for s in seed_list:
#         random.seed(s)
#         idx_all = np.arange(len(ally))
#         idx_train = []
#         idx_val = []
#         for i in range(ally.shape[1]):
#             curr_label_idx = np.where(ally[:, i] == 1)[0]
#             r = random.sample(range(0, len(curr_label_idx)), 20)
#             idx_train.append(curr_label_idx[r].tolist())
#         idx_train = np.array(idx_train).reshape(-1)

#         for j in idx_train:
#             idx_all = np.delete(idx_all, np.where(idx_all == j)[0])
#         idx_val = np.copy(idx_all)
#         idx_train_list.append(idx_train)
#         idx_val_list.append(idx_val)

    for s in seed_list:
        np.random.seed(s)
        random.seed(s)
        idx_all = np.arange(len(ally))
        idx_train = []
        for i in range(ally.shape[1]):
            curr_label_idx = np.where(ally[:, i] == 1)[0]
            r = random.sample(range(0, len(curr_label_idx)), 20)
            idx_train.append(curr_label_idx[r].tolist())
        idx_train = np.array(idx_train).reshape(-1)
        np.random.shuffle(idx_train)

        # balance for val
        for j in idx_train:
            idx_all = np.delete(idx_all, np.where(idx_all == j)[0])
        idx_val_all = np.copy(idx_all)
        label_val = np.argmax(labels[idx_val_all], 1)
        num = np.ceil(500 / labels.shape[1]).astype(int)
        idx_val = []
        for i in range(np.max(label_val)+1):
            #print(i)
            curr_label_idx = np.where(label_val == i)[0]
            idx_val.append(np.random.choice(curr_label_idx, size=num, replace=False).tolist())
        idx_val = np.array(idx_val).reshape(-1)
        np.random.shuffle(idx_val)

#         idx_val = np.random.choice(idx_val, size=500, replace=False)

        #print(idx_train, idx_train.shape)
        #print(idx_val, idx_val.shape)
        #sys.exit()
        
        idx_train_list.append(idx_train)
        idx_val_list.append(idx_val)
        
    return h, features, edge_features, labels, idx_train_list, idx_val_list, idx_test

def load_data_sparse_trials_ratio(dataset_str, num_train):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    h = adj.copy()
    
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    edge_features = get_edge_features_sparse(h, features)
    
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    
    #if dataset_str != 'citeseer':
    features = normalize_features(features)
    edge_features = normalize_features(edge_features)
    
    idx_test = test_idx_range
    seed_list = np.arange(100)
    idx_train_list = []
    idx_val_list = []

    for s in seed_list:
        np.random.seed(s)
        random.seed(s)
        idx_all = np.arange(len(ally))
        idx_train = []
        for i in range(ally.shape[1]):
            curr_label_idx = np.where(ally[:, i] == 1)[0]
            r = random.sample(range(0, len(curr_label_idx)), num_train)
            idx_train.append(curr_label_idx[r].tolist())
        idx_train = np.array(idx_train).reshape(-1)
        np.random.shuffle(idx_train)

        # balance for val
        for j in idx_train:
            idx_all = np.delete(idx_all, np.where(idx_all == j)[0])
        idx_val_all = np.copy(idx_all)
        label_val = np.argmax(labels[idx_val_all], 1)
        num = np.ceil(500 / labels.shape[1]).astype(int)
        idx_val = []
        for i in range(np.max(label_val)+1):
            #print(i)
            curr_label_idx = np.where(label_val == i)[0]
            idx_val.append(np.random.choice(curr_label_idx, size=num, replace=False).tolist())
        idx_val = np.array(idx_val).reshape(-1)
        np.random.shuffle(idx_val)
        
        idx_train_list.append(idx_train)
        idx_val_list.append(idx_val)
        
    return h, features, edge_features, labels, idx_train_list, idx_val_list, idx_test
"""

class TrainSet(data.Dataset):
    def __init__(self, num_item, num_negative, train_data, train_mat):
        super().__init__()
        self.num_item = num_item
        self.num_ng = num_negative
        self.train_data = train_data
        self.train_mat = train_mat

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        user, pos_item = self.train_data[idx]
        all_item = [pos_item]
        while len(all_item) <= self.num_ng:
            j = np.random.randint(self.num_item)
            while (user, j) in self.train_mat or j in all_item:
                j = np.random.randint(self.num_item)
            all_item.append(j)
        return torch.LongTensor([user]), torch.LongTensor(all_item)


class TestSet(data.Dataset):
    def __init__(self, num_item, num_negative, test_data, train_mat, seed=123):
        super().__init__()
        self.num_item = num_item
        self.num_ng = num_negative
        self.test_data = test_data
        self.train_mat = train_mat
        np.random.seed(seed)
        self._ng_sample()

    def _ng_sample(self):
        self.users = []
        self.items = []
        for x in self.test_data:
            u, i = x[0], x[1]
            all_item = [i]
            for t in range(self.num_ng):
                j = np.random.randint(self.num_item)
                while (u, j) in self.train_mat or j == i:
                    j = np.random.randint(self.num_item)
                all_item.append(j)
            self.users.append([u])
            self.items.append(all_item)

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        user = self.users[idx]
        item = self.items[idx]
        return torch.LongTensor(user), torch.LongTensor(item)
    
    
    
def get_dataloader(train_data, val_data, test_data, data_mat, setting):
    train_set = TrainSet(setting.num_item, setting.num_negative, train_data, data_mat)
    val_set = TestSet(setting.num_item, setting.num_negative_eval, val_data, data_mat, seed=213)
    test_set = TestSet(setting.num_item, setting.num_negative_eval, test_data, data_mat)

    train_loader = DataLoader(train_set, batch_size=setting.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=setting.batch_size_eval, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=setting.batch_size_eval, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader


def load_all(train_path, val_path, test_path):
    train_data = pd.read_csv(train_path,
        sep=',', header=None, names=['user', 'item'],
        usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
    train_data = train_data.values.tolist()

    val_data = pd.read_csv(val_path,
        sep=',', header=None, names=['user', 'item'],
        usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
    val_data = val_data.values.tolist()

    test_data = pd.read_csv(test_path,
        sep=',', header=None, names=['user', 'item'],
        usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
    test_data = test_data.values.tolist()

    return train_data, val_data, test_data


def manage(num_user, num_item, train_data, val_data):
    train_mat = sp.dok_matrix((num_user, num_item), dtype=np.float32)

    for x in train_data:
        train_mat[x[0], x[1]] = 1.0

    for x in val_data:
        train_mat[x[0], x[1]] = 1.0

    return train_mat


def load_data_mat(dataset_str, num_user, num_item):
    data_mat = scio.loadmat("./data/{}/{}.mat".format(dataset_str, dataset_str))

    h = data_mat['H']
    X = data_mat['X']
    Y = data_mat['Y']
    # Y = nn.Parameter(
    #         torch.FloatTensor(h.shape[1], X.shape[1]), requires_grad=True)
    # Y = nn.init.xavier_normal_(Y)
    # Y = Y.detach().numpy()
    
    train_path = "./data/{}/data.train".format(dataset_str)
    val_path = "./data/{}/data.val".format(dataset_str)
    test_path = "./data/{}/data.test".format(dataset_str)
    train_data, val_data, test_data = load_all(train_path, val_path, test_path)
    M = manage(num_user, num_item, train_data, val_data)   

    # M = data_mat['M']   
    # train_data = data_mat['train_data'].tolist()
    # val_data = data_mat['val_data'].tolist()
    # test_data = data_mat['test_data'].tolist()
    
    X = normalize_features(X)
    Y = normalize_features(Y)
    
    return h, X, Y, M, train_data, val_data, test_data


# def load_data_mat_error_bound(dataset_str):
#     data_mat = scio.loadmat("data/{}.mat".format(dataset_str))
#     h = data_mat['h']
#     X = data_mat['X']
#     Y = data_mat['Y']
#     labels = data_mat['labels']
#     idx_train_list = data_mat['idx_train_list']
#     idx_val_list = data_mat['idx_val_list']
    
# #     print(train.shape)
# #     print(val.shape)
# #     sys.exit()
# #     indices = []
    
#     X = normalize_features(X)
#     Y = normalize_features(Y)
    
#     return h, X, Y, labels, idx_train_list, idx_val_list



def load_data_mat_bmc(dataset_str):
    data_mat = scio.loadmat("data/{}.mat".format(dataset_str))
    h = sp.csr_matrix(data_mat['H']).tolil()
    X = data_mat['X']
    Y = data_mat['Y']
    idx_r_train_list = data_mat['idx_r_train_list']
    idx_c_train_list = data_mat['idx_c_train_list']
    idx_r_test_list = data_mat['idx_r_test_list']
    idx_c_test_list = data_mat['idx_c_test_list']
    
    X = normalize_features(X)
    Y = normalize_features(Y)
    
    return h, X, Y, idx_r_train_list, idx_c_train_list, idx_r_test_list, idx_c_test_list

def load_data_mat_clustering(dataset_str):
    data_mat = scio.loadmat("data/{}.mat".format(dataset_str))
    h = data_mat['h']
    X = data_mat['X']
    Y = data_mat['Y']
    labels = data_mat['labels']
#     idx_train_list = data_mat['idx_train_list']
#     idx_val_list = data_mat['idx_val_list']
    
#     X = normalize_features(X)
#     Y = normalize_features(Y)
    
    return h, X, Y, labels
'''
def load_data_mat_mc(DATASET, FEATURES, TESTING):
    if DATASET == 'ml_1m' or DATASET == 'ml_100k' or DATASET == 'douban':
        NUMCLASSES = 5
    elif DATASET == 'ml_10m':
        NUMCLASSES = 10
        print('\n WARNING: this might run out of RAM, consider using train_minibatch.py for dataset %s' % DATASET)
        print('If you want to proceed with this option anyway, uncomment this.\n')
        sys.exit(1)
    elif DATASET == 'flixster':
        NUMCLASSES = 10
    elif DATASET == 'yahoo_music':
        NUMCLASSES = 71
        ACCUM = 'stack'
        if ACCUM == 'sum':
            print('\n WARNING: combining DATASET=%s with ACCUM=%s can cause memory issues due to large number of classes.')
            print('Consider using "--accum stack" as an option for this dataset.')
            print('If you want to proceed with this option anyway, uncomment this.\n')
            sys.exit(1)

    # Splitting dataset in training, validation and test set
    DATASEED = 1234
    SPLITFROMFILE = True
    VERBOSE = True
    if DATASET == 'ml_1m' or DATASET == 'ml_10m':
        if FEATURES:
            datasplit_path = 'data/' + DATASET + '/withfeatures_split_seed' + str(DATASEED) + '.pickle'
        else:
            datasplit_path = 'data/' + DATASET + '/split_seed' + str(DATASEED) + '.pickle'
    elif FEATURES:
        datasplit_path = 'data/' + DATASET + '/withfeatures.pickle'
    else:
        datasplit_path = 'data/' + DATASET + '/nofeatures.pickle'


    if DATASET == 'flixster' or DATASET == 'douban' or DATASET == 'yahoo_music':
        u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices, \
            val_labels, val_u_indices, val_v_indices, test_labels, \
            test_u_indices, test_v_indices, class_values, rating, rating_train, rating_test = load_data_monti(DATASET, TESTING)

    elif DATASET == 'ml_100k':
        print("Using official MovieLens dataset split u1.base/u1.test with 20% validation set size...")
        u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices, \
            val_labels, val_u_indices, val_v_indices, test_labels, \
            test_u_indices, test_v_indices, class_values, binary_rating, rating = load_official_trainvaltest_split(DATASET, TESTING)
    else:
        print("Using random dataset split ...")
        u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices, \
            val_labels, val_u_indices, val_v_indices, test_labels, \
            test_u_indices, test_v_indices, class_values, binary_rating, rating = create_trainvaltest_split(DATASET, DATASEED, TESTING,
                                                                                     datasplit_path, SPLITFROMFILE,
                                                                                     VERBOSE)

    num_users, num_items = adj_train.shape

    num_side_features = 0
    
#     print(val_u_indices, test_u_indices)
#     print(val_u_indices.shape, test_u_indices.shape)
#     sys.exit()
    

    # feature loading
    if not FEATURES:
        u_features = sp.identity(num_users, format='csr')
        v_features = sp.identity(num_items, format='csr')

        u_features, v_features = preprocess_user_item_features(u_features, v_features)

    elif FEATURES and u_features is not None and v_features is not None:
        # use features as side information and node_id's as node input features

        print("Normalizing feature vectors...")
        u_features_side = normalize_features(u_features)
        v_features_side = normalize_features(v_features)

        u_features_side, v_features_side = preprocess_user_item_features(u_features_side, v_features_side)
        
        u_features_side = np.array(u_features_side.todense(), dtype=np.float32)
        v_features_side = np.array(v_features_side.todense(), dtype=np.float32)

        num_side_features = u_features_side.shape[1]

        # node id's for node input features
        id_csr_v = sp.identity(num_items, format='csr')
        id_csr_u = sp.identity(num_users, format='csr')

        u_features, v_features = preprocess_user_item_features(id_csr_u, id_csr_v)

#         u_features = normalize_features(u_features)
#         v_features = normalize_features(v_features)

    else:
        raise ValueError('Features flag is set to true but no features are loaded from dataset ' + DATASET)
    
#     adj_test = np.zeros((adj_train.shape))
#     for i in range(adj_test.shape[0]):
#         adj_test[test_u_indices[i], test_v_indices[i]] = test_labels[i]
#     print(test_labels.shape)
#     print(adj_test)
#     sys.exit()
    
    mask_train = adj_train.copy()
    mask_train[mask_train != 0] = 1
    
    """
    # Collect all user and item nodes for test set
    test_u = list(set(test_u_indices))
    test_v = list(set(test_v_indices))
    test_u_dict = {n: i for i, n in enumerate(test_u)}
    test_v_dict = {n: i for i, n in enumerate(test_v)}

    test_u_indices = np.array([test_u_dict[o] for o in test_u_indices])
    test_v_indices = np.array([test_v_dict[o] for o in test_v_indices])


    # Collect all user and item nodes for validation set
    val_u = list(set(val_u_indices))
    val_v = list(set(val_v_indices))
    val_u_dict = {n: i for i, n in enumerate(val_u)}
    val_v_dict = {n: i for i, n in enumerate(val_v)}

    val_u_indices = np.array([val_u_dict[o] for o in val_u_indices])
    val_v_indices = np.array([val_v_dict[o] for o in val_v_indices])


    # Collect all user and item nodes for train set
    train_u = list(set(train_u_indices))
    train_v = list(set(train_v_indices))
    train_u_dict = {n: i for i, n in enumerate(train_u)}
    train_v_dict = {n: i for i, n in enumerate(train_v)}

    train_u_indices = np.array([train_u_dict[o] for o in train_u_indices])
    train_v_indices = np.array([train_v_dict[o] for o in train_v_indices])
    """
    
#     print(train_u_indices.shape)
#     print(val_u_indices.shape)
#     print(test_u_indices.shape)
#     sys.exit()
    
    return binary_rating, rating, mask_train, train_labels, train_u_indices, train_v_indices, val_labels, val_u_indices, val_v_indices, test_labels, test_u_indices, test_v_indices, class_values, u_features, v_features
'''
    
    

def load_data_mat_link(dataset_str):
    dataset = scio.loadmat("data/{}.mat".format(dataset_str))
    
    h = dataset['h']
    X = dataset['X']
    Y = dataset['Y']
    h_train = dataset['h_train']
    idx_test_list = dataset['idx_test_list']
    idx_test_false_list = dataset['idx_test_false_list']

    
    X = normalize_features(X)
    Y = normalize_features(Y)
    
    return h, h_train, X, Y, idx_test_list, idx_test_false_list

def load_data_mat_rec(dataset_str):
    dataset = scio.loadmat("data/{}.mat".format(dataset_str))
    
    h = dataset['h']
    h_train = dataset['h_train']
    test_label = dataset['test_label']
    
    return h, h_train, test_label

"""
def load_data_link(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        '''
        fix Pickle incompatibility of numpy arrays between Python 2 and 3
        https://stackoverflow.com/questions/11305790/pickle-incompatibility-of-numpy-arrays-between-python-2-and-3
        '''
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as rf:
            u = pkl._Unpickler(rf)
            u.encoding = 'latin1'
            cur_data = u.load()
            objects.append(cur_data)
        # objects.append(
        #     pkl.load(open("data/ind.{}.{}".format(dataset, names[i]), 'rb')))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    
    return adj, features
"""

def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false
    
def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    
    if np.where(rowsum == 0)[0].shape[0] != 0:
        indices = np.where(rowsum == 0)[0]
        for i in indices:
            rowsum[i] = float('inf')
    
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    if np.where(rowsum == 0)[0].shape[0] != 0:
        indices = np.where(rowsum == 0)[0]
        for i in indices:
            rowsum[i] = float('inf')
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def kernelized_features(x, y):
    
    # sparse
    k1 = sp.hstack([x, sp.csr_matrix((x.shape[0], y.shape[1]))])
    k2 = sp.hstack([sp.csr_matrix((y.shape[0], x.shape[1])), y])
    k = sp.vstack([k1, k2])
    k = normalize_features(np.dot(k, k.transpose()))
    x = k[np.arange(0, k1.shape[0]), :]
    y = k[np.arange(k1.shape[0], k.shape[0]), :]
    
#     k1 = np.hstack((x, np.zeros((x.shape[0], y.shape[1]))))
#     k2 = np.hstack((np.zeros((y.shape[0], x.shape[1])), y))
#     k = np.vstack((k1, k2))
#     k = normalize_features(np.dot(k, k.transpose()))
#     x = k[np.arange(0, k1.shape[0]), :]
#     y = k[np.arange(k1.shape[0], k.shape[0]), :]
#     print(x, y)
#     sys.exit()
    return x, y

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels) 

def accuracy_lp(output, labels):
    
    preds = torch.zeros(labels.shape).cuda()
    preds[output > 0.5] = 1
    
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels) 

'''
metrics for model performance 
'''
def link_accuracy(output, labels, numMissing):
    results = {}

    preds = output.max(1)[1].type_as(labels)
    preds = preds.cpu().data.numpy()
    labels = labels.cpu().data.numpy()

    scores = output[:, 0]
    scores = scores.cpu().data.numpy()
    
    sortedIndices = np.argsort(scores)
    sortedIndices = sortedIndices[-numMissing:]
    
    indices = [i for i,x in enumerate(preds) if x == 0 and x == labels[i] and i in sortedIndices]
    results['p'] = len(indices)
#     results['p'] = len(indices)/numMissing

    output = output.cpu().data.numpy()
    y = np.array(labels)

    pred = np.array(output[:,1]) 
    results['auc'] = metrics.roc_auc_score(y, pred)

    return results

def get_roc_score(adj_rec, adj_orig, edges_pos, edges_neg):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    #adj_rec = np.dot(x, y.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
#         preds_neg.append(adj_rec[e[0], e[1]])
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

def patk(actual, pred, k):
    #we return 0 if k is 0 because 
    #   we can't divide the no of common values by 0 
    if k == 0:
        return 0

    #taking only the top k predictions in a class 
    k_pred = pred[:k]

    #taking the set of the actual values 
    actual_set = set(actual)

    #taking the set of the predicted values 
    pred_set = set(k_pred)

    #taking the intersection of the actual set and the pred set
        # to find the common values
    common_values = actual_set.intersection(pred_set)

    return len(common_values)/len(pred[:k])

def apatk(acutal, pred, k):
    #creating a list for storing the values of precision for each k 
    precision_ = []
    for i in range(1, k+1):
        #calculating the precision at different values of k 
        #      and appending them to the list 
        precision_.append(pk.patk(acutal, pred, i))

    #return 0 if there are no values in the list
    if len(precision_) == 0:
        return 0 

    #returning the average of all the precision values
    return np.mean(precision_)

def mapk(acutal, pred, k):
    #creating a list for storing the Average Precision Values
    average_precision = []
    #interating through the whole data and calculating the apk for each 
    for i in range(len(acutal)):
        average_precision.append(apk.apatk(acutal[i], pred[i], k))

    #returning the mean of all the data
    return np.mean(average_precision)

def ml_accuracy(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        temp += sum(np.logical_and(y_true[i], y_pred[i])) / sum(np.logical_or(y_true[i], y_pred[i]))
    return temp / y_true.shape[0]

def hamming_Loss(y_true, y_pred):
    temp=0
    for i in range(y_true.shape[0]):
        temp += np.size(y_true[i] == y_pred[i]) - np.count_nonzero(y_true[i] == y_pred[i])
    return temp/(y_true.shape[0] * y_true.shape[1])
    

def calculate_metrics(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=float)
    
#     sum_pred = np.sum(pred, 1)
#     print(np.where(sum_pred == 0))
#     sys.exit()
    
    return {
        'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro'),
            'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro'),
            'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro'),
            'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro'),
            'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro'),
            'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro'),
            'samples/precision': precision_score(y_true=target, y_pred=pred, average='samples'),
            'samples/recall': recall_score(y_true=target, y_pred=pred, average='samples'),
            'samples/f1': f1_score(y_true=target, y_pred=pred, average='samples'),
            'mr': np.all(pred == target, axis=1).mean(),
            '01Loss': np.any(target != pred, axis=1).mean(),
            'accuracy': ml_accuracy(target, pred),
            'hammingLoss': hamming_Loss(target, pred)
            }


def normalize_sparse_hypergraph_symmetric(H):
    
    rowsum = np.array(H.sum(1))
    #print(rowsum.shape)
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    D = sp.diags(r_inv_sqrt)
    
    colsum = np.array(H.sum(0))
    #print(colsum.shape)
    r_inv_sqrt = np.power(colsum, -1).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    B = sp.diags(r_inv_sqrt)
    
    Omega = sp.eye(B.shape[0])

    hx1 = D.dot(H).dot(Omega).dot(B).dot(H.transpose()).dot(D)
    hx2 = D.dot(H).dot(Omega).dot(B)

    return hx1, hx2


def normalize_sparse_hypergraph_symmetric2(H):
    
    rowsum = np.array(H.sum(1))
    #print(rowsum.shape)

    if np.where(rowsum == 0)[0].shape[0] != 0:
        indices = np.where(rowsum == 0)[0]
        for i in indices:
            rowsum[i] = float('inf')
    
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    D = sp.diags(r_inv_sqrt)
    

    colsum = np.array(H.sum(0))
    #print(colsum.shape)
    
    if np.where(colsum == 0)[0].shape[0] != 0:
        indices = np.where(colsum == 0)[0]
        for i in indices:
            colsum[i] = float('inf')
    
    r_inv_sqrt = np.power(colsum, -1).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    B = sp.diags(r_inv_sqrt)
    
    Omega = sp.eye(B.shape[0])


    hx1 = D.dot(H).dot(Omega).dot(B).dot(H.transpose()).dot(D)
    hx2 = D.dot(H).dot(Omega).dot(B)

    return hx1, hx2    
    
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)    
        
        
class clustering_metrics():
    def __init__(self, true_label, predict_label):
        self.true_label = true_label
        self.pred_label = predict_label


    def clusteringAcc(self):
        # best mapping between true_label and predict label
        l1 = list(set(self.true_label))
        numclass1 = len(l1)

        l2 = list(set(self.pred_label))
        numclass2 = len(l2)
        if numclass1 != numclass2:
            print('Class Not equal, Error!!!!')
            return 0

        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]

                cost[i][j] = len(mps_d)

        # match two clustering results by Munkres algorithm
        m = Munkres()
        cost = cost.__neg__().tolist()

        indexes = m.compute(cost)

        # get the match results
        new_predict = np.zeros(len(self.pred_label))
        for i, c in enumerate(l1):
            # correponding label in l2:
            c2 = l2[indexes[i][1]]

            # ai is the index with label==c2 in the pred_label list
            ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
            new_predict[ai] = c

        acc = metrics.accuracy_score(self.true_label, new_predict)
        f1_macro = metrics.f1_score(self.true_label, new_predict, average='macro')
        precision_macro = metrics.precision_score(self.true_label, new_predict, average='macro')
        recall_macro = metrics.recall_score(self.true_label, new_predict, average='macro')
        f1_micro = metrics.f1_score(self.true_label, new_predict, average='micro')
        precision_micro = metrics.precision_score(self.true_label, new_predict, average='micro')
        recall_micro = metrics.recall_score(self.true_label, new_predict, average='micro')
        return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro

    def evaluationClusterModelFromLabel(self):
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        adjscore = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro = self.clusteringAcc()

#         print('ACC=%f, f1_macro=%f, precision_macro=%f, recall_macro=%f, f1_micro=%f, precision_micro=%f, recall_micro=%f, NMI=%f, ADJ_RAND_SCORE=%f' % (acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, nmi, adjscore))

#         fh = open('recoder.txt', 'a')

#         fh.write('ACC=%f, f1_macro=%f, precision_macro=%f, recall_macro=%f, f1_micro=%f, precision_micro=%f, recall_micro=%f, NMI=%f, ADJ_RAND_SCORE=%f' % (acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, nmi, adjscore) )
#         fh.write('\r\n')
#         fh.flush()
#         fh.close()

        return acc, nmi, f1_macro, precision_macro, recall_macro, adjscore
        
        