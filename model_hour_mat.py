import torch.nn as nn
from layer_hour import GraphConvolution_, GraphConvolution
import torch.nn.functional as F
import torch


class HOUR_Mat(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, activation, neg_slope, X_nodes, Y_nodes, num_user, num_item, lam, device):
        super(HOUR_Mat, self).__init__()

        self.device = device

        self.l_u = nn.Linear(num_item, nfeat)
        self.l_i = nn.Linear(num_user, nfeat)
        self.l_e = nn.Linear(nfeat, nfeat)

        self.gc1 = GraphConvolution_(nfeat, nhid, 1, activation, neg_slope)
        self.gc2 = GraphConvolution(nhid, nclass, 1)

        self.dropout = dropout
        self.activation = activation
        self.neg_slope = neg_slope
        self.X_nodes = X_nodes
        self.Y_nodes = Y_nodes
        self.num_user = num_user
        self.num_item = num_item
        self.lam = lam

    def predict(self, users_feature, items_feature):
        return torch.sum(users_feature * items_feature, dim=2)
    
    def regularize(self, users_feature, items_feature):
        '''
        embeddings of targets for predicting -> extra loss(default: L2 loss...)
        '''
        loss = self.lam * \
               ((users_feature ** 2).sum() + (items_feature ** 2).sum())
        return loss

    def forward(self, M, HHT, H, HT, HTH, H_all, useridx, itemidx):
        x_u = self.l_u(M)
        x_u = F.relu(x_u)
        x_u = F.dropout(x_u, self.dropout, training=self.training)
        x_i = self.l_i(M.T)
        x_i = F.relu(x_i)
        x_i = F.dropout(x_i, self.dropout, training=self.training)
        x = torch.cat((x_u, x_i))
        y = torch.mm(H_all.T,x)
        y = self.l_e(y)
        y = F.relu(y)
        y = F.dropout(y, self.dropout, training=self.training)

        x_cat = [x]

        HHT_1 = self.gc1(x, HHT)
        H_1 = self.gc1(y, H)
        HT_1 = self.gc1(x, HT)
        HTH_1 = self.gc1(y, HTH)
        HHT_1 = F.dropout(HHT_1, self.dropout, training=self.training)
        H_1 = F.dropout(H_1, self.dropout, training=self.training)
        HT_1 = F.dropout(HT_1, self.dropout, training=self.training)
        HTH_1 = F.dropout(HTH_1, self.dropout, training=self.training)

        # x = HHT_1 + H_1
        # y = HT_1 + HTH_1
        x = HHT_1*0.3 + H_1*1
        y = HT_1*1 + HTH_1*1
        
        x_cat += [x]
        HHT_2 = self.gc2(x, HHT)
        H_2 = self.gc2(y, H)
        HT_2 = self.gc2(x, HT)
        HTH_2 = self.gc2(y, HTH)
        # HHT_2 = F.leaky_relu(HHT_2, negative_slope=self.neg_slope)
        # H_2 = F.leaky_relu(H_2, negative_slope=self.neg_slope)
        # HT_2 = F.leaky_relu(HT_2, negative_slope=self.neg_slope)
        # HTH_2 = F.leaky_relu(HTH_2, negative_slope=self.neg_slope)
        HHT_2 = F.relu(HHT_2)
        H_2 = F.relu(H_2)
        HT_2 = F.relu(HT_2)
        HTH_2 = F.relu(HTH_2)

        # x = HHT_2 + H_2
        # y = HT_2 + HTH_2
        x = HHT_2*0.3 + H_2*1
        y = HT_2*1 + HTH_2*1
        reconstruct = torch.cat([x, y], dim=0)
        x_cat += [x]

        # get recovered feat
        hat_x = reconstruct[0:self.X_nodes]
        hat_y = reconstruct[self.X_nodes : self.X_nodes + self.Y_nodes]
        recovered_H = torch.mm(hat_x, hat_y.t())
        recovered_H = torch.sigmoid(recovered_H)      # shape=heyperEdge feat shape = (u+i)*egde

        ### get user item embedding
        x_cat = torch.cat(x_cat,dim=1)
        users_feat = x_cat[:self.num_user]
        items_feat = x_cat[self.num_user:]
        items_embedding = items_feat[itemidx]              # (bs, item_num=9, embdims=96) items_emb.shape=(bs,9,embdim)
        users_embedding = users_feat[useridx].expand(
            - 1, itemidx.shape[1], -1)

        pred = self.predict(users_embedding, items_embedding)   # (bs, item_num=9)

        reg_loss = self.regularize(users_embedding, items_embedding)

        ### get reconstruct M
        rec_M = torch.mm(x_u,x_i.T)
        # rec_M = torch.sigmoid(torch.mm(x_u,x_i.T))
        rec_M_mask = torch.mul(rec_M, M)
        rec_loss = F.binary_cross_entropy_with_logits(rec_M_mask, M)

        ### get pred from x_embd
        pred_label = torch.zeros((pred.shape)).to(self.device)
        pred_label[:,0] = 1
        bce_loss = F.binary_cross_entropy_with_logits(pred, pred_label)

        # ### get pred from recons_M
        # user_idx = useridx.expand(-1, itemidx.shape[1])
        # p_idx = torch.stack((user_idx, itemidx), dim=2)
        # pred_M = rec_M[p_idx]

        return recovered_H, pred, reg_loss, rec_loss, bce_loss