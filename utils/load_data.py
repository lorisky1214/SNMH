import numpy as np
import scipy.sparse as sp
import torch as th
from sklearn.preprocessing import OneHotEncoder


def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot

def preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)

def load_acm(type_num):
    path = "data/acm/"

    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    label = th.FloatTensor(label)

    nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    nei_s = np.load(path + "nei_s.npy", allow_pickle=True)
    pa_num = 0
    for i, j in enumerate(nei_a):
        pa_num += nei_a[i].shape[0]
    ps_num = 0
    for i, j in enumerate(nei_s):
        ps_num += nei_s[i].shape[0]
    nei_a = [th.LongTensor(i) for i in nei_a]
    nei_s = [th.LongTensor(i) for i in nei_s]

    feat_p = sp.load_npz(path + "p_feat.npz")
    feat_a = sp.eye(type_num[1])
    feat_s = sp.eye(type_num[2])
    feat_p = th.FloatTensor(preprocess_features(feat_p))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_s = th.FloatTensor(preprocess_features(feat_s))

    pap = sp.load_npz(path + "pap.npz")
    psp = sp.load_npz(path + "psp.npz")
    pap_num = pap.count_nonzero()
    psp_num = psp.count_nonzero()
    pap = sparse_mx_to_torch_sparse_tensor(normalize_adj(pap))
    psp = sparse_mx_to_torch_sparse_tensor(normalize_adj(psp))

    train = np.load(path + "train.npy")
    test = np.load(path + "test.npy")
    val = np.load(path + "val.npy")
    train = th.LongTensor(train)
    test = th.LongTensor(test)
    val = th.LongTensor(val)

    return [nei_a, nei_s], [pa_num, ps_num], [feat_p, feat_a, feat_s], [pap, psp], \
           [pap_num, psp_num], label, train, val, test

def load_dblp(type_num):
    path = "data/dblp/"

    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    label = th.FloatTensor(label)

    nei_p = np.load(path + "nei_p.npy", allow_pickle=True)
    ap_num = 0
    for i, j in enumerate(nei_p):
        ap_num += nei_p[i].shape[0]
    nei_p = [th.LongTensor(i) for i in nei_p]

    feat_a = sp.load_npz(path + "a_feat.npz").astype("float32")
    feat_p = sp.eye(type_num[1])
    feat_p = th.FloatTensor(preprocess_features(feat_p))
    feat_a = th.FloatTensor(preprocess_features(feat_a))

    apa = sp.load_npz(path + "apa.npz")
    apcpa = sp.load_npz(path + "apcpa.npz")
    aptpa = sp.load_npz(path + "aptpa.npz")
    apa_num = apa.count_nonzero()
    apcpa_num = apcpa.count_nonzero()
    aptpa_num = aptpa.count_nonzero()
    apa = sparse_mx_to_torch_sparse_tensor(normalize_adj(apa))
    apcpa = sparse_mx_to_torch_sparse_tensor(normalize_adj(apcpa))
    aptpa = sparse_mx_to_torch_sparse_tensor(normalize_adj(aptpa))

    train = np.load(path + "train.npy")
    test = np.load(path + "test.npy")
    val = np.load(path + "val.npy")
    train = th.LongTensor(train)
    test = th.LongTensor(test)
    val = th.LongTensor(val)

    return [nei_p], [ap_num], [feat_a, feat_p], [apa, apcpa, aptpa], \
           [apa_num, apcpa_num, aptpa_num], label, train, val, test

def load_freebase(type_num):
    path = "data/freebase/"

    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    label = th.FloatTensor(label)

    nei_d = np.load(path + "nei_d.npy", allow_pickle=True)
    nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    nei_w = np.load(path + "nei_w.npy", allow_pickle=True)
    md_num = 0
    for i, j in enumerate(nei_d):
        md_num += nei_d[i].shape[0]
    ma_num = 0
    for i, j in enumerate(nei_a):
        ma_num += nei_a[i].shape[0]
    mw_num = 0
    for i, j in enumerate(nei_w):
        mw_num += nei_w[i].shape[0]
    nei_d = [th.LongTensor(i) for i in nei_d]
    nei_a = [th.LongTensor(i) for i in nei_a]
    nei_w = [th.LongTensor(i) for i in nei_w]

    feat_m = sp.eye(type_num[0])
    feat_d = sp.eye(type_num[1])
    feat_a = sp.eye(type_num[2])
    feat_w = sp.eye(type_num[3])
    feat_m = th.FloatTensor(preprocess_features(feat_m))
    feat_d = th.FloatTensor(preprocess_features(feat_d))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_w = th.FloatTensor(preprocess_features(feat_w))

    mam = sp.load_npz(path + "mam.npz")
    mdm = sp.load_npz(path + "mdm.npz")
    mwm = sp.load_npz(path + "mwm.npz")
    mam_num = mam.count_nonzero()
    mdm_num = mdm.count_nonzero()
    mwm_num = mwm.count_nonzero()
    mam = sparse_mx_to_torch_sparse_tensor(normalize_adj(mam))
    mdm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mdm))
    mwm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mwm))

    train = np.load(path + "train.npy")
    test = np.load(path + "test.npy")
    val = np.load(path + "val.npy")
    train = th.LongTensor(train)
    test = th.LongTensor(test)
    val = th.LongTensor(val)

    return [nei_d, nei_a, nei_w], [md_num, ma_num, md_num], [feat_m, feat_d, feat_a, feat_w], \
           [mdm, mam, mwm], [mdm_num, mam_num, mwm_num], label, train, val, test

def load_data(dataset, type_num):
    if dataset == "acm":
        data = load_acm(type_num)
    elif dataset == "dblp":
        data = load_dblp(type_num)
    elif dataset == "freebase":
        data = load_freebase(type_num)
    return data


















