import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import NMF

from utils_parser import parse_args


def load_all(data_path):
    data = pd.read_csv(data_path,
        sep=',', header=None, names=['user', 'item'],
        usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
    data = data.values.tolist()

    return data


def manage(data, args):
    data_mat = sp.dok_matrix((args.num_user, args.num_item), dtype=np.float32)

    for x in data:
        data_mat[x[0], x[1]] = 1.0

    return data_mat


def prepare_data(args):
    train_data = load_all(args.train_path)
    data_mat = manage(train_data, args)
    return data_mat


def write_txt(info, save_path):
    line = ''
    with open(save_path,'w') as file:
        for i in info:
            for e in range(len(i)):
                line=line+str(i[e])+' '
            file.write(line.strip(' '))
            file.write('\n')
            line=''
    file.close()


def construct_hyperedge(mat, save_path):
    res = []
    for u_id in range(mat.shape[0]):   # Traverse each user/item to find the hyperedge that is most associated with it.
        hyper_res = [u_id]
        hyper_w = mat[u_id, :]
        hyper_sort = np.flip(np.argsort(hyper_w))   # Sort by weight in descending order
        for hyper_id in hyper_sort:
            if hyper_w[hyper_id]!=0:
                hyper_res.append(hyper_id)       # Save the hyperedges with non-zero weights
        
        res.append(hyper_res)
        
    write_txt(res, save_path)



def matrix_factorization(data_mat, emb_num):
    nmf = NMF(n_components=emb_num)  
    user_distribution = nmf.fit_transform(data_mat)  
    item_distribution = nmf.components_

    return user_distribution, item_distribution
    


def get_topk_edges(topk, path, save_path):
    with open(path, 'r') as f:
        res = []
        for line in f.readlines():
            line_res = []
            line=line.strip().split('\t')
            values_list = line[0].split(' ')
            for i in range(min(topk+1, len(values_list))):
                line_res.append(int(values_list[i]))
            res.append(line_res)
    f.close()
    write_txt(res, save_path)   


def get_social_relation(path):
    edge = {}
    with open(path, 'r') as f:
        res = []
        for line in f.readlines():
            line=line.strip().split('\t')
            values_list = line[0].split(' ')
            uid = int(values_list[0])
            for i in range(1, len(values_list)):
                edge_id = values_list[i]
                if edge_id:
                    if int(edge_id) in edge.keys():
                        edge[int(edge_id)].append(uid)
                    else:
                        edge[int(edge_id)]=[uid]
    f.close()

    for ix, val in edge.items():
        edge[ix] = list(set(val))

    edge = dict(sorted(edge.items(), key=lambda item: item[0]))

    return edge


def get_inter_edge(social_user, social_item, user_save_path, item_save_path):
    user_e = []
    item_e = []
    for eid, value in social_user.items():
        user_e.append(eid)

    for eid, value in social_item.items():
        item_e.append(eid)

    hyper_e = [i for i in user_e if i in item_e]

    res = []
    for eid, value in social_user.items():
        if eid in hyper_e:
            values_list = [eid] + list(value)
            res.append(values_list)

    for eid in range(len(hyper_e)):  # edgeid is sorted again from 0.
        res[eid][0] = eid
    write_txt(res, user_save_path)

    res = []
    for eid, value in social_item.items():
        if eid in hyper_e:
            values_list = [eid] + list(value)
            res.append(values_list)
            
    for eid in range(len(hyper_e)):  # edgeid is sorted again from 0.
        res[eid][0] = eid
    write_txt(res, item_save_path)


if __name__ == '__main__':
    args = parse_args()
    # Beidian user=3773 item=4544
    emb_num = 1886 

    data_mat = prepare_data(args)
    user_distribution, item_distribution = matrix_factorization(data_mat, emb_num)
    item_distribution_T = pd.DataFrame(item_distribution).transpose().values

    # get the relationships of each user/item and edge
    construct_hyperedge(user_distribution, args.hyperedge_user_train_path)  # (userid, edgeid0, edgeid1, ...)
    construct_hyperedge(item_distribution_T, args.hyperedge_item_train_path)  # (itemid, edgeid0, edgeid1, ...)

    # each user/item only retains the topK most closely connected edges. 
    topk = args.top_k
    get_topk_edges(topk, args.hyperedge_user_train_path, args.hyperedge_user_topk_train_path)  # (userid, edgeid0, edgeid1, ...)
    get_topk_edges(topk, args.hyperedge_item_train_path, args.hyperedge_item_topk_train_path)  # (itemid, edgeid0, edgeid1, ...)

    # using "edge" as the index, obtain the users_id/items_id associated with each edge.
    social_user = get_social_relation(args.hyperedge_user_topk_train_path)  # (edgeid, userid0, userid1, ...)
    social_item = get_social_relation(args.hyperedge_item_topk_train_path)  # (edgeid, itemid0, itemid1, ...)

    # extract the edges of the intersection set of social_user and social_item, and renumber them starting from 0.
    get_inter_edge(social_user, social_item, args.social_user_train_path, args.social_item_train_path)


