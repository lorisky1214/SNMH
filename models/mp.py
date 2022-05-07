import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np


class MP(nn.Module):
    def __init__(self, hidden_dim, pred_hid, feats_dim_list, mps_num,
                 feat_drop, attn_drop, P, agg, moving_average_decay=0.99, epochs=1000):
        super(MP, self).__init__()
        self.target_feat_trans = nn.Linear(feats_dim_list[0], hidden_dim, bias=True)
        nn.init.xavier_normal_(self.target_feat_trans.weight, gain=1.414)
        if feat_drop > 0:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        self.P = P
        self.hidden_dim = hidden_dim
        self.encoder = nn.ModuleList([GCN(hidden_dim, hidden_dim) for _ in range(P)])
        self.stu_idx = mps_num.index(max(mps_num, key=abs))
        for i in range(P):
            if i != self.stu_idx:
                set_requires_grad(self.encoder[i], False)
        self.teacher_ema_updater = EMA(moving_average_decay, epochs)
        self.student_predictor = nn.Sequential(nn.Linear(hidden_dim, pred_hid),
                                               nn.BatchNorm1d(pred_hid, momentum=0.01),
                                               nn.PReLU(),
                                               nn.Linear(pred_hid, hidden_dim))
        self.student_predictor.apply(init_weights)
        self.agg = agg
        if self.agg == 1:
            self.fuse = type_level_att(hidden_dim, attn_drop)

    def forward(self, feats, mps):
        h = F.elu(self.feat_drop(self.target_feat_trans(feats[0])))
        v_student = self.encoder[self.stu_idx](h, mps[self.stu_idx])
        v_pred = self.student_predictor(v_student)
        v_teacher = []
        with torch.no_grad():
            for i in range(self.P):
                if i != self.stu_idx:
                    v_teacher.append(self.encoder[i](h, mps[i]))
        loss = []
        for i in range(len(v_teacher)):
            loss.append(loss_fn(v_pred, v_teacher[i].detach()))
        loss_sum = 0
        for i in range(len(loss)):
            temp = loss[i].mean()
            loss_sum += temp
        loss_final = loss_sum / len(loss)
        v_teacher.append(v_student)
        z_sum = v_teacher
        if self.agg == 0:
            z_out = 0
            for i in range(self.P):
                z_out += z_sum[i]
            z_out /= self.P
        else:
            z_out = self.fuse(z_sum)
        return z_out, loss_final

    def get_embeds(self, feats, mps):
        h = F.elu(self.target_feat_trans(feats[0]))
        z_sum = []
        for i in range(self.P):
            z_sum.append(self.encoder[i](h, mps[i]))
        if self.agg == 0:
            z_out = 0
            for i in range(self.P):
                z_out += z_sum[i]
            z_out /= self.P
        else:
            z_out = self.fuse(z_sum)
        return z_out.detach()

    def update_moving_average(self):
        update_moving_average(self.teacher_ema_updater, self.encoder, self.stu_idx, self.P)

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=1.414)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj):
        seq_fts = self.fc(seq)
        out = torch.spmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)

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

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

def update_moving_average(ema_updater, encoder, stu_idx, P):
    for i in range(P):
        if i != stu_idx:
            for stu_params, tea_params in zip(encoder[stu_idx].parameters(), encoder[i].parameters()):
                old_weight, up_weight = tea_params.data, stu_params.data
                tea_params.data = ema_updater.update_average(old_weight, up_weight)