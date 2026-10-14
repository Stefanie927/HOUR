import pandas as pd
import numpy as np
import scipy.sparse as sp
import scipy.io as scio
import torch
import torch.nn as nn
from utils_parser import parse_args

args = parse_args()

### Beidian
user_num = args.num_user     # Beidian user=3773 item=4544 edge=1782
item_num = args.num_item
hyperedge_num = 1782   
save_path = args.dataset_save_path

# construct the hyperedge matrix H
user_path = args.social_user_train_path
item_path = args.social_item_train_path
H = sp.dok_matrix((user_num+item_num, hyperedge_num), dtype=np.float32)

with open(user_path, 'r') as f:
    for line in f.readlines():
        line=line.strip().split('\t')
        values_list = line[0].split(' ')
        eid = int(values_list[0])
        for uid in values_list[1:]:
            H[int(uid), eid] = 1.0
f.close()

with open(item_path, 'r') as f:
    for line in f.readlines():
        line=line.strip().split('\t')
        values_list = line[0].split(' ')
        eid = int(values_list[0])
        for iid in values_list[1:]:
            H[int(user_num+int(iid)), eid] = 1.0
f.close()


# construct the scoring matrix M
train_data = pd.read_csv(args.train_path,
    sep=',', header=None, names=['user', 'item'],
    usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
train_data = train_data.values.tolist()

val_data = pd.read_csv(args.val_path,
    sep=',', header=None, names=['user', 'item'],
    usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
val_data = val_data.values.tolist()

test_data = pd.read_csv(args.test_path,
    sep=',', header=None, names=['user', 'item'],
    usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
test_data = test_data.values.tolist()


M = sp.dok_matrix((user_num, item_num), dtype=np.float32)

for x in train_data:
    M[x[0], x[1]] = 1.0


# construct node embedding
emb_dim = 32
X = nn.Parameter(
            torch.FloatTensor(user_num+item_num, emb_dim), requires_grad=True)
X = nn.init.xavier_normal_(X)
X = X.detach().numpy()

# construct edge embedding
Y = H.T.dot(X)


scio.savemat(save_path, {'H': H, 'X': X, 'Y': Y, 'M': M,
                        'train_data': train_data,
                        'val_data': val_data, 
                        'test_data': test_data})