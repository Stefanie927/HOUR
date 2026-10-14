import numpy as np
from utils import load_data_mat, dotdict, seed_everything, accuracy, normalize_sparse_hypergraph_symmetric, get_dataloader
import sys
import time
from model_hour_mat import HOUR_Mat
import torch
import torch.nn.functional as F
from torch import nn, optim
import os
import argparse
import csv
import scipy.sparse as sp
import scipy.io as scio


class Evaluate:
    '''
        designed for leave-one-out test-set
    '''

    def recall(self, rlist):
        hit = torch.sum(rlist)
        return hit / rlist.shape[0]

    def dcg(self, rlist):
        dcg_list = torch.arange(2, rlist.shape[1] + 2, dtype=torch.float32, device=rlist.device)
        dcg_list = torch.log2(dcg_list).pow(-1)
        self.rdcg = rlist * dcg_list

    def ndcg(self, k):
        rdcg = self.rdcg[:, :k]
        ndcg = torch.sum(rdcg, dim=-1)
        return torch.mean(ndcg)

    def evaluate(self, pred, topK):
        '''
        topK : [1, 3, 5, 10]
        pred : shape = batch_size * ( 1 + num_neg)
        '''
        metrics = dict()
        pred[:, 0] = pred[:, 0] - torch.finfo(pred.dtype).tiny

        topK = sorted(topK, reverse=True)
        _, rlist = torch.topk(pred, topK[0])
        rlist = (rlist == 0).double()

        self.dcg(rlist)
        for k in topK:
            rlist = rlist[:, :k]
            recall = self.recall(rlist)
            ndcg = self.ndcg(k)
            metrics['ndcg@%d' % k] = ndcg.item()
            metrics['recall@%d' % k] = recall.item()
        return metrics
    


class _Loss(nn.Module):
    def __init__(self, reduction='mean'):
        '''
        `reduction` (string, optional)
        - Specifies the reduction to apply to the output: `none` | `mean` | `sum`. `none`: no reduction will be applied, `mean`: the sum of the output will be divided by the number of elements in the output, `sum`: the output will be summed. Note: size_average and reduce are in the process of being deprecated, and in the meantime, specifying either of those two args will override reduction. Default: `sum`
        '''
        super().__init__()
        assert (reduction == 'mean' or reduction == 'sum' or reduction == 'none')
        self.reduction = reduction


class BPRLoss(_Loss):
    def __init__(self, reduction='mean'):
        '''
        `reduction` (string, optional)
        - Specifies the reduction to apply to the output: `none` | `mean` | `sum`. `none`: no reduction will be applied, `mean`: the sum of the output will be divided by the number of elements in the output, `sum`: the output will be summed. Note: size_average and reduce are in the process of being deprecated, and in the meantime, specifying either of those two args will override reduction. Default: `sum`
        '''
        super().__init__(reduction)

    def forward(self, pred, reg_loss, batch_size=None):
        '''
        `model_output` (tensor) - column 0 must be the scores of positive pois, column 1 must be the negative.
        '''
        neg_pred = pred[:, 1:]
        pos_pred = pred[:, 0].unsqueeze(-1).expand(-1, neg_pred.shape[1])
        e = 1e-10
        loss = - torch.log(torch.sigmoid(pos_pred - neg_pred)+e)
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        else:
            raise ValueError("reduction must be 'mean' | 'sum'")
        if batch_size is not None and self.reduction == 'mean':
            loss += reg_loss / batch_size
        else:
            loss += reg_loss
        return loss



def training(data, args, train_loader, val_loader, test_loader, s = 2021):

    seed_everything(seed = s)

    H_trainX = torch.from_numpy(data.H_trainX.toarray()).float().to(args.device)
    X = torch.from_numpy(data.X).float().to(args.device)
    Y = torch.from_numpy(data.Y).float().to(args.device)
    M = torch.from_numpy(data.M.toarray()).float().to(args.device)

    HHT = torch.from_numpy(data.HHT.toarray()).float().to(args.device)
    H = torch.from_numpy(data.H.toarray()).float().to(args.device)
    HTH = torch.from_numpy(data.HTH.toarray()).float().to(args.device)
    HT = torch.from_numpy(data.HT.toarray()).float().to(args.device)

    gamma1 = args.gamma1
    gamma2 = args.gamma2
    gamma3 = args.gamma3
    epochs = args.epochs
    learning_rate = args.learning_rate
    dropout = args.dropout
    activation = args.activation

    x_n_nodes = X.shape[0]
    y_n_nodes = Y.shape[0]
    pos_weight = float(H_trainX.shape[0] * H_trainX.shape[0] - H_trainX.sum()) / H_trainX.sum()
    norm = H_trainX.shape[0] * H_trainX.shape[0] / float((H_trainX.shape[0] * H_trainX.shape[0] - H_trainX.sum()) * 2)

    loss_func = BPRLoss()

    eval_func = Evaluate()
    metrics_key = []
    for k in eval(args.topK):
        metrics_key.append('recall@%d' % k)
        metrics_key.append('ndcg@%d' % k)
    
    model = HOUR_Mat(args.dim_hidden0, args.dim_hidden, args.dim_hidden2, dropout, activation, args.weight, args.num_node, args.num_edge, args.num_user, args.num_item, args.lam, args.device)
    model.to(args.device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    best_epoch = 0
    best_test_recall = 0
    for epoch in range(epochs):
        model.train()

        tmp_train_loss = 0
        num_train_batch = len(train_loader)
        for user_idx, item_idx in train_loader:   # user_idx: user idx list, item_idx: items idx list
            user_idx, item_idx = user_idx.to(args.device), item_idx.to(args.device)
            optimizer.zero_grad()

            recovered, pred, reg_loss, rec_loss, bce_loss = model(M, HHT, H, HT, HTH, H_trainX, user_idx, item_idx)  
            loss1 = loss_func(pred, reg_loss, args.batch_size)
            loss2 = F.binary_cross_entropy_with_logits(recovered, H_trainX, pos_weight=pos_weight)
            loss_train = loss1+ gamma1*loss2 + gamma2*rec_loss + gamma3*bce_loss
            tmp_train_loss += loss_train.item()
            
            loss_train.backward()
            optimizer.step()

        perf_str = "Epoch: {:d} | ".format(epoch)
        perf_str += 'train loss: {:.4f} | '.format(tmp_train_loss/num_train_batch)


        # val        
        with torch.no_grad():
            model.eval()
            tmp_val_loss = 0
            num_val_batch = len(val_loader)
            all_val_pred = []
            for user_idx, item_idx in val_loader:
                user_idx, item_idx = user_idx.to(args.device), item_idx.to(args.device)
                recovered, pred, reg_loss, rec_loss, bce_loss = model(M, HHT, H, HT, HTH, H_trainX, user_idx, item_idx)   
                loss = loss_func(pred, reg_loss, args.batch_size_eval)
                tmp_val_loss += loss.item()
                all_val_pred += [pred]

            all_val_pred = torch.cat(all_val_pred, dim=0)
            metrics = eval_func.evaluate(all_val_pred, eval(args.topK))
        
        # Test
        with torch.no_grad():
            model.eval()

            tmp_test_loss = 0
            num_test_batch = len(test_loader)
            all_test_pred = []
            for user_idx, item_idx in test_loader:
                user_idx, item_idx = user_idx.to(args.device), item_idx.to(args.device) 
                recovered, pred, reg_loss, rec_loss, bce_loss = model(M, HHT, H, HT, HTH, H_trainX, user_idx, item_idx) 
                loss = loss_func(pred, reg_loss, args.batch_size_eval)
                tmp_test_loss += loss.item()
                all_test_pred += [pred]

            all_test_pred = torch.cat(all_test_pred, dim=0)
            metrics = eval_func.evaluate(all_test_pred, eval(args.topK))
            perf_str += 'test loss: {:.4f} | '.format(tmp_test_loss/num_test_batch)
            perf_str += "test metrics: "
            for mk in metrics_key:
                perf_str = perf_str + mk + '=%.4f, ' % metrics[mk]

        if metrics['recall@10'] >= best_test_recall:
            best_test_recall = metrics['recall@10']
            best_epoch = epoch
        torch.save(model.state_dict(), './results/model_{}.pth'.format(epoch))

        print (perf_str)

    print(best_epoch, best_test_recall)

    return perf_str



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='HGAE')
    parser.add_argument('--gpu_id', type=str, default='0', help="device id to run")
    parser.add_argument('--dataname', type=str, default='Beidian', help="dataname to run")
    parser.add_argument('--num_item', type=int, default=4544, help="Beidian")
    parser.add_argument('--num_user', type=int, default=3773, help="Beidian")
    parser.add_argument('--num_negative', type=int, default=8, help="neg item")   
    parser.add_argument('--num_negative_eval', type=int, default=100, help="neg item")   
    parser.add_argument('--batch_size', type=int, default=4096, help="batch_size")         # 512 4096  
    parser.add_argument('--batch_size_eval', type=int, default=4096, help="batch_size")    # 512 4096
    parser.add_argument('--topK', default='[1,3,5,10]', help="top k")
    setting = parser.parse_args()
    
    device = torch.cuda.current_device()
    
    H, X, Y, M, train_data, val_data, test_data = load_data_mat(setting.dataname, setting.num_user, setting.num_item)
    train_loader, val_loader, test_loader = get_dataloader(train_data, val_data, test_data, M, setting)
        
    H_trainX = H.copy()

    HHT, H = normalize_sparse_hypergraph_symmetric(H_trainX)                  # H.shape=(u+i)*e=8317*1782   HHT.shape=8317*8317   HT.shape=1782*8317
    HTH, HT = normalize_sparse_hypergraph_symmetric(H_trainX.transpose())                                 # HTH.shape=1782*1782         HT.shape=1782*8317

    if setting.dataname == "citeseer1000":
        dim_hidden = 512
        learning_rate = 0.001
        weight_decay = 0.001
        gamma = 10
#         alpha = 0.1
#         beta = 0.1
        alpha = 0.8
        beta = 0.2
    elif setting.dataname == "pubmed1000":
        dim_hidden = 512
        learning_rate = 0.01
        weight_decay = 0.0001
        gamma = 0.1
        alpha = 0.9
        beta = 0.1
    elif setting.dataname == "cora1000":
        dim_hidden = 512
        learning_rate = 0.01
        weight_decay = 0.0001
        gamma = 1
        alpha = 0.1
        beta = 0.1
    elif setting.dataname == "Beidian": 
        dim_hidden0 = 2048
        dim_hidden = 1024    
        dim_hidden2 = 512   
        lam = 1e-5    
        learning_rate = 1e-4   
        weight_decay = 0.0001
        gamma1 = 0.1  # 1
        gamma2 = 20
        gamma3 = 0.1
        alpha = 0.1
        beta = 0.1
    
    
    epochs = 200
    seed = 2021
    early = 100
    dropout = 0
    # activation = 'leaky_relu'
    activation = 'relu'
    neg_slope = 0.2
    weight = 1
        
    data = dotdict()
    args = dotdict()

    data.X = X
    data.Y = Y
    data.H_trainX = H_trainX
    data.HHT = HHT
    data.H = H
    data.HT = HT
    data.HTH = HTH
    data.M = M

    args.dim_hidden0 = dim_hidden0
    args.dim_hidden = dim_hidden
    args.dim_hidden2 = dim_hidden2
    args.weight_decay = weight_decay
    args.epochs = epochs
    args.early_stop = early
    args.learning_rate = learning_rate
    args.dropout = dropout
    args.activation = activation
    args.neg_slope = neg_slope
    args.weight = weight
    args.gamma1 = gamma1
    args.gamma2 = gamma2
    args.gamma3 = gamma3
    args.alpha = alpha
    args.beta = beta
    args.verb = 0
    args.verb2 = 0
    args.num_user = setting.num_user
    args.num_item = setting.num_item
    args.num_node = X.shape[0]
    args.num_edge = Y.shape[0]
    args.batch_size = setting.batch_size
    args.batch_size_eval = setting.batch_size_eval
    args.topK = setting.topK
    args.device = device
    args.lam = lam

    perf_str = training(data, args, train_loader, val_loader, test_loader, s=seed)
    
    print(perf_str)


