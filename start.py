'''
! Name: start.py
! Author: Cziun
! Last Update Date: 2022-05-04

Function: Start the training of SNMH
'''

import numpy as np
import torch
import warnings
import datetime
import random
from torch import optim
from torch.backends import cudnn
from utils import load_data, set_params, evaluate
from models import SNMH

# ignore warnings
warnings.filterwarnings('ignore')

args = set_params()
if torch.cuda.is_available():
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device("cpu")

# Set global seed to get steady result
seed = args.seed
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
cudnn.benchmark = False
cudnn.deterministic = True

# Train SNMH
def train():
    nei_index, rel_num, feats, mps, mps_num, label, idx_train, idx_val, idx_test = \
        load_data(args.dataset, args.type_num)

    nb_classes = label.shape[-1]
    feats_dim_list = [i.shape[1] for i in feats]
    P = int(len(mps))

    model = SNMH(args.hidden_dim, args.pred_hid, feats_dim_list, mps_num,
               args.feat_drop, args.attn_drop, P, rel_num, args.agg, args.sample_rate, args.nei_num)
    SNMH_optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)
    SNMH_scheduler = lambda epoch: epoch / 1000 if epoch < 1000 \
        else (1 + np.cos((epoch - 1000) * np.pi / (args.nb_epochs - 1000))) * 0.5
    SNMH_scheduler = optim.lr_scheduler.LambdaLR(SNMH_optimiser, lr_lambda=SNMH_scheduler)

    if torch.cuda.is_available():
        print('Using CUDA')
        model = model.to(device)
        feats = [feat.cuda() for feat in feats]
        mps = [mp.cuda() for mp in mps]
        label = label.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    cnt_wait = 0
    best = 1e9
    best_t = 0
    start_time = datetime.datetime.now()
    for epoch in range(0, args.nb_epochs):
        model.train()
        SNMH_optimiser.zero_grad()
        loss = model(args.lam, feats, mps, nei_index)
        print("loss: ", loss.data.cpu())
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'SNMH_' + args.dataset + '.pkl')
        else:
            cnt_wait += 1
        if cnt_wait == args.patience:
            print('Early stopping!')
            break
        loss.backward()
        SNMH_optimiser.step()
        SNMH_scheduler.step()
        model.mp.update_moving_average()
        model.rt.update_moving_average()
    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('SNMH_' + args.dataset + '.pkl'))
    model.eval()
    embeds = model.get_embeds(feats, mps)
    evaluate(embeds, idx_train, idx_val, idx_test, label, nb_classes, device, args.eva_lr, args.eva_wd)
    end_time = datetime.datetime.now()
    time = (end_time - start_time).seconds
    print("Total time: ", time, "s")

    if args.save_emb:
        np.save("embeds_" + args.dataset, embeds.cpu().numpy())

if __name__ == '__main__':
    train()