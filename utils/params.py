import argparse
import sys


argv = sys.argv
dataset = argv[1]

def acm_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', type=bool, default=True)
    parser.add_argument('--dataset', type=str, default="acm")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=70)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--pred_hid', type=int, default=128)
    parser.add_argument('--nb_epochs', type=int, default=10000)
    parser.add_argument('--eva_lr', type=float, default=0.05)
    parser.add_argument('--eva_wd', type=float, default=0)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.5)
    parser.add_argument('--l2_coef', type=float, default=0)
    parser.add_argument('--feat_drop', type=float, default=0.2)
    parser.add_argument('--attn_drop', type=float, default=0.3)
    parser.add_argument('--sample_rate', nargs='+', type=int, default=[7, 1])
    parser.add_argument('--lam', type=float, default=0)
    parser.add_argument('--agg', type=int, default=0)  # 0: avg  1: sem_level_att
    args, _ = parser.parse_known_args()
    args.type_num = [4019, 7167, 60]  # p a s
    args.nei_num = 2  # the num of 1-hop neighbor types
    return args

def dblp_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', type=bool, default=True)
    parser.add_argument('--dataset', type=str, default="dblp")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=23)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--pred_hid', type=int, default=128)
    parser.add_argument('--nb_epochs', type=int, default=10000)
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.5)
    parser.add_argument('--l2_coef', type=float, default=0)
    parser.add_argument('--feat_drop', type=float, default=0.4)
    parser.add_argument('--attn_drop', type=float, default=0.35)
    parser.add_argument('--sample_rate', nargs='+', type=int, default=[6])
    parser.add_argument('--lam', type=float, default=0.8)
    parser.add_argument('--agg', type=int, default=1)  # 0: avg  1: sem_level_att
    args, _ = parser.parse_known_args()
    args.type_num = [4057, 14328, 7723, 20]  # a p t c
    args.nei_num = 1  # the num of 1-hop neighbor types
    return args

def freebase_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', type=bool, default=True)
    parser.add_argument('--dataset', type=str, default="freebase")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=45)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--pred_hid', type=int, default=128)
    parser.add_argument('--nb_epochs', type=int, default=10000)
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.5)
    parser.add_argument('--l2_coef', type=float, default=0)
    parser.add_argument('--feat_drop', type=float, default=0.1)
    parser.add_argument('--attn_drop', type=float, default=0.3)
    parser.add_argument('--sample_rate', nargs='+', type=int, default=[2, 18, 3])
    parser.add_argument('--lam', type=float, default=0)
    parser.add_argument('--agg', type=int, default=0)  # 0: avg  1: sem_level_att
    args, _ = parser.parse_known_args()
    args.type_num = [3492, 2502, 33401, 4459]  # m, d, a, w
    args.nei_num = 3  # the num of 1-hop neighbor types
    return args

def set_params():
    if dataset == "acm":
        args = acm_params()
    elif dataset == "dblp":
        args = dblp_params()
    elif dataset == "freebase":
        args = freebase_params()
    return args