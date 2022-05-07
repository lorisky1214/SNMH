import torch.nn as nn
import torch.nn.functional as F
import torch
from .mp import MP
from .rt import RT


class SNMH(nn.Module):
    def __init__(self, hidden_dim, pred_hid, feats_dim_list, mps_num,
                 feat_drop, attn_drop, P, rel_num, agg, sample_rate, nei_num):
        super(SNMH, self).__init__()
        self.mp = MP(hidden_dim, pred_hid, feats_dim_list, mps_num, feat_drop, attn_drop, P, agg)
        self.rt = RT(hidden_dim, feats_dim_list, feat_drop, attn_drop, rel_num, sample_rate, nei_num)
        self.predictor = nn.Sequential(nn.Linear(hidden_dim, pred_hid),
                                       nn.BatchNorm1d(pred_hid, momentum=0.01),
                                       nn.PReLU(),
                                       nn.Linear(pred_hid, hidden_dim))
        self.predictor.apply(init_weights)

    def forward(self, lam, feats, mps, nei_index):
        h_mp, loss_cv = self.mp(feats, mps)
        z_mp = self.predictor(h_mp)
        h_rt = self.rt(feats, nei_index)
        loss_cs = loss_fn(z_mp, h_rt)
        loss = lam * loss_cv + (1 - lam) * loss_cs.mean()
        return loss

    def get_embeds(self, feats, mps):
        h_mp = self.mp.get_embeds(feats, mps)
        embeds = h_mp
        return embeds.detach()


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)