import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np


class RT(nn.Module):
    def __init__(self, hidden_dim, feats_dim_list, feat_drop, attn_drop,
                 rel_num, sample_rate, nei_num, moving_average_decay=0.99, epochs=1000):
        super(RT, self).__init__()
        self.hidden_dim = hidden_dim
        self.nei_num = nei_num
        self.sample_rate = sample_rate
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True)
                                      for feats_dim in feats_dim_list])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        if feat_drop > 0:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        self.encoder = nn.ModuleList([node_level_att(hidden_dim, attn_drop)
                                      for _ in range(nei_num)])
        self.stu_idx = rel_num.index(max(rel_num, key=abs))
        for i in range(nei_num):
            if i != self.stu_idx:
                set_requires_grad(self.encoder[i], False)
        self.teacher_ema_updater = EMA(moving_average_decay, epochs)
        self.fuse = type_level_att(hidden_dim, attn_drop)

    def forward(self, feats, nei_index):
        h_all = []
        for i in range(len(feats)):
            h_all.append(F.elu(self.feat_drop(self.fc_list[i](feats[i]))))
        sele_nei_stu = []
        for per_node_nei in nei_index[self.stu_idx]:
            if len(per_node_nei) >= self.sample_rate[self.stu_idx]:
                select_one = torch.tensor(np.random.choice(per_node_nei, self.sample_rate[self.stu_idx],
                                                           replace=False))[np.newaxis]
            else:
                select_one = torch.tensor(np.random.choice(per_node_nei, self.sample_rate[self.stu_idx],
                                                           replace=True))[np.newaxis]
            sele_nei_stu.append(select_one)
        sele_nei_stu = torch.cat(sele_nei_stu, dim=0)
        if torch.cuda.is_available():
            sele_nei_stu = sele_nei_stu.cuda()
        v_student = F.elu(self.encoder[self.stu_idx](sele_nei_stu, h_all[self.stu_idx + 1], h_all[0]))
        v_teacher = []
        with torch.no_grad():
            for i in range(self.nei_num):
                if i != self.stu_idx:
                    sele_nei = []
                    sample_num = self.sample_rate[i]
                    for per_node_nei in nei_index[i]:
                        if len(per_node_nei) >= sample_num:
                            select_one = torch.tensor(np.random.choice(per_node_nei, sample_num,
                                                                       replace=False))[np.newaxis]
                        else:
                            select_one = torch.tensor(np.random.choice(per_node_nei, sample_num,
                                                                       replace=True))[np.newaxis]
                        sele_nei.append(select_one)
                    sele_nei = torch.cat(sele_nei, dim=0)
                    if torch.cuda.is_available():
                        sele_nei = sele_nei.cuda()
                    one_type_emb = F.elu(self.encoder[i](sele_nei, h_all[i + 1], h_all[0]))
                    v_teacher.append(one_type_emb)
        v_teacher.append(v_student)
        z_sum = v_teacher
        z_out = self.fuse(z_sum)
        return z_out

    def update_moving_average(self):
        update_moving_average(self.teacher_ema_updater, self.encoder, self.stu_idx, self.nei_num)


class node_level_att(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(node_level_att, self).__init__()
        self.att = nn.Parameter(torch.empty(size=(1, 2 * hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x
        self.softmax = nn.Softmax(dim=1)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, nei, nei_h, target_h):
        nei_emb = F.embedding(nei, nei_h)
        target_h = torch.unsqueeze(target_h, 1)
        target_h = target_h.expand_as(nei_emb)
        all_emb = torch.cat([target_h, nei_emb], dim=-1)
        attn_curr = self.attn_drop(self.att)
        att = self.leakyrelu(all_emb.matmul(attn_curr.t()))
        att = self.softmax(att)
        nei_emb = (att * nei_emb).sum(dim=1)
        return nei_emb

class type_level_att(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(type_level_att, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)
        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)
        self.softmax = nn.Softmax()
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        z_mc = 0
        for i in range(len(embeds)):
            z_mc += embeds[i] * beta[i]
        return z_mc

class EMA:
    def __init__(self, beta, epochs):
        super().__init__()
        self.beta = beta
        self.step = 0
        self.total_steps = epochs

    def update_average(self, old, new):
        if old is None:
            return new
        beta = 1 - (1 - self.beta) * (np.cos(np.pi * self.step / self.total_steps) + 1) / 2.0
        self.step += 1
        return old * beta + (1 - beta) * new

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

def update_moving_average(ema_updater, encoder, stu_idx, nei_num):
    for i in range(nei_num):
        if i != stu_idx:
            for stu_params, tea_params in zip(encoder[stu_idx].parameters(), encoder[i].parameters()):
                old_weight, up_weight = tea_params.data, stu_params.data
                tea_params.data = ema_updater.update_average(old_weight, up_weight)