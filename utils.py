import datetime
import numpy as np
import os
import os.path as osp
import random
import logging
from pathlib import Path

from scipy import sparse
from scipy import io as sio
from itertools import product

import torch
import dgl

def set_random_seed(seed=0):
    random.seed(seed)  # python
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # pytorch
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # pytorch-cuda


def get_date_postfix():
    dt = datetime.datetime.now()
    post_fix = '{}_{:02d}-{:02d}-{:02d}'.format(dt.date(), dt.hour, dt.minute, dt.second)
    return post_fix


def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size, dtype=torch.bool)
    mask[indices] = 1
    return mask.byte()


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp

spllll = 100

def load_acm_raw(train_split=0.2, val_split=0.3, feat=1):
    data_folder = './data/acm/'
    data_path = osp.join(data_folder, 'ACM.mat')
    data = sio.loadmat(data_path)
    target = 'paper'  ###

    """
    {0: ('paper', 'cite', 'paper'),
    1: ('paper', 'ref', 'paper'),
    2: ('paper', 'to', 'author'),
    3: ('author', 'to', 'paper'),
    4: ('paper', 'to', 'subject'),
    5: ('subject', 'to', 'paper'),
    6: ('paper', 'to', 'term'),
    7: ('term', 'to', 'paper')}
    """
    # paper author subject term
    # TvsP PvsA PvsV AvsF VvsC PvsL PvsC PvsT
    p_vs_l = data['PvsL']
    p_vs_a = data['PvsA']
    p_vs_p = data['PvsP']

    p_vs_t = data['PvsT']
    p_vs_c = data['PvsC']
    # (7245, 1)	1.0
    conf_ids = [0, 1, 9, 10, 13]
    label_ids = [0, 1, 2, 2, 1]

    p_vs_c_filter = p_vs_c[:, conf_ids]
    p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]
    p_vs_c = p_vs_c[p_selected]
    p_vs_p = p_vs_p[p_selected].T[p_selected]
    a_selected = (p_vs_a[p_selected].sum(0) != 0).A1.nonzero()[0]
    p_vs_a = p_vs_a[p_selected].T[a_selected].T
    l_selected = (p_vs_l[p_selected].sum(0) != 0).A1.nonzero()[0]
    p_vs_l = p_vs_l[p_selected].T[l_selected].T
    t_selected = (p_vs_t[p_selected].sum(0) != 0).A1.nonzero()[0]
    p_vs_t = p_vs_t[p_selected].T[t_selected].T

    if feat == 1 or feat == 3:
        hg = dgl.heterograph({
            ('paper', 'paper_paper_cite', 'paper'): p_vs_p.nonzero(),
            ('paper', 'paper_paper_ref', 'paper'): p_vs_p.transpose().nonzero(),
            # ptp
            # ('paper', 'paper_term', 'term'): p_vs_t.nonzero(),
            # ('term', 'term_paper', 'paper'): p_vs_t.transpose().nonzero(),
            # pap
            ('paper', 'paper_author', 'author'): p_vs_a.nonzero(),
            ('author', 'author_paper', 'paper'): p_vs_a.transpose().nonzero(),
            # psp
            ('paper', 'paper_subject', 'subject'): p_vs_l.nonzero(),
            ('subject', 'subject_paper', 'paper'): p_vs_l.transpose().nonzero(),
        })
        hg_HAN = dgl.heterograph({
            ('paper', 'paper_paper_cite', 'paper'): p_vs_p.nonzero(),
            ('paper', 'paper_paper_ref', 'paper'): p_vs_p.transpose().nonzero(),
            # ptp
            # ('paper', 'paper_term', 'term'): p_vs_t.nonzero(),
            # ('term', 'term_paper', 'paper'): p_vs_t.transpose().nonzero(),
            # pap
            ('paper', 'paper_author', 'author'): p_vs_a.nonzero(),
            ('author', 'author_paper', 'paper'): p_vs_a.transpose().nonzero(),
            # psp
            ('paper', 'paper_subject', 'subject'): p_vs_l.nonzero(),
            ('subject', 'subject_paper', 'paper'): p_vs_l.transpose().nonzero(),
        })

        paper_feats = torch.FloatTensor(p_vs_t.toarray())
        # features = paper_feats
        features = {
            'paper': paper_feats
        }
    elif feat == 2:
        hg = dgl.heterograph({
            ('paper', 'paper_paper_cite', 'paper'): p_vs_p.nonzero(),
            ('paper', 'paper_paper_ref', 'paper'): p_vs_p.transpose().nonzero(),
            ('paper', 'paper_author', 'author'): p_vs_a.nonzero(),
            ('author', 'author_paper', 'paper'): p_vs_a.transpose().nonzero(),
            ('paper', 'paper_subject', 'subject'): p_vs_l.nonzero(),
            ('subject', 'subject_paper', 'paper'): p_vs_l.transpose().nonzero(),
            ('paper', 'paper_term', 'term'): p_vs_t.nonzero(),
            ('term', 'term_paper', 'paper'): p_vs_t.transpose().nonzero()
        })
        features = {}
    elif feat == 4:
        hg = dgl.heterograph({
            ('paper', 'paper_paper_cite', 'paper'): p_vs_p.nonzero(),
            ('paper', 'paper_paper_ref', 'paper'): p_vs_p.transpose().nonzero(),
            ('paper', 'paper_author', 'author'): p_vs_a.nonzero(),
            ('author', 'author_paper', 'paper'): p_vs_a.transpose().nonzero(),
            ('paper', 'paper_subject', 'subject'): p_vs_l.nonzero(),
            ('subject', 'subject_paper', 'paper'): p_vs_l.transpose().nonzero(),
        })

        paper_feats = torch.FloatTensor(p_vs_t.toarray())
        features = {}
    print(hg)
    print(hg_HAN)

    pc_p, pc_c = p_vs_c.nonzero()
    labels = np.zeros(len(p_selected), dtype=np.int64)
    for conf_id, label_id in zip(conf_ids, label_ids):
        labels[pc_p[pc_c == conf_id]] = label_id
    labels = torch.LongTensor(labels)

    num_classes = 3

    # conf_ids = [0, 1, 9, 10, 13]
    # float_mask = np.zeros(len(pc_p))
    # for conf_id in conf_ids:
    #     pc_c_mask = pc_c == conf_id
    #     float_mask[pc_c_mask] = np.random.permutation(
    #         np.linspace(0, 1, pc_c_mask.sum())
    #     )
    # train_idx = np.where(float_mask <= 0.2)[0]
    # val_idx = np.where((float_mask > 0.2) & (float_mask <= 0.3))[0]
    # test_idx = np.where(float_mask > 0.3)[0]


    split = np.load(osp.join(data_folder, f'train_val_test_idx_{int(train_split * spllll)}_2021.npz'))
    # split = np.load(osp.join(data_folder, f'train_val_test_idx_{int(train_split * 100)}_2021.npz'))
    train_idx = split['train_idx']
    val_idx = split['val_idx']
    test_idx = split['test_idx']

    num_nodes = hg.number_of_nodes('paper')
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    # float_mask = np.zeros(len(pc_p))
    # for conf_id in conf_ids:
    #     pc_c_mask = pc_c == conf_id
    #     float_mask[pc_c_mask] = np.random.permutation(
    #         np.linspace(0, 1, pc_c_mask.sum())
    #     )
    # train_idx = np.where(float_mask <= 0.2)[0]
    # val_idx = np.where((float_mask > 0.2) & (float_mask <= 0.3))[0]
    # test_idx = np.where(float_mask > 0.3)[0]
    #
    # num_nodes = hg.number_of_nodes("paper")
    # train_mask = get_binary_mask(num_nodes, train_idx)
    # val_mask = get_binary_mask(num_nodes, val_idx)
    # test_mask = get_binary_mask(num_nodes, test_idx)

    node_dict = {}
    edge_dict = {}
    for ntype in hg.ntypes:
        node_dict[ntype] = len(node_dict)
    for etype in hg.etypes:
        edge_dict[etype] = len(edge_dict)
        hg.edges[etype].data['id'] = torch.ones(hg.number_of_edges(etype), dtype=torch.long) * edge_dict[etype]
        hg_HAN.edges[etype].data['id'] = torch.ones(hg_HAN.number_of_edges(etype), dtype=torch.long) * edge_dict[etype]

    return hg,hg_HAN, node_dict, edge_dict,paper_feats, features, labels, num_classes, train_idx, val_idx, \
           test_idx, train_mask, val_mask, test_mask, target


def load_dblp_precessed(train_split=0.2, val_split=0.3, feat=1):
    raw_dir = './data/dblp/'

    author_feats = sparse.load_npz(osp.join(raw_dir, 'features_0.npz')) # author to keyword
    paper_feats = sparse.load_npz(osp.join(raw_dir, 'features_1.npz')) # paper to words in title
    term_feats = np.load(osp.join(raw_dir, 'features_2.npy'))
    node_type_idx = np.load(osp.join(raw_dir, 'node_types.npy'))
    target = 'author'

    if feat == 1:
        author_feats = torch.from_numpy(author_feats.todense()).to(torch.float)
        paper_feats = torch.from_numpy(paper_feats.todense()).to(torch.float)
        term_feats = torch.from_numpy(term_feats).to(torch.float)
        features = {
            'author': author_feats,
            'paper': paper_feats,
            'term': term_feats
        }
    elif feat == 2 or feat == 4:
        features = {}
    elif feat == 3:
        author_feats = torch.from_numpy(author_feats.todense()).to(torch.float)
        features = {
            'author': author_feats,
        }
    node_type_idx = torch.from_numpy(node_type_idx).to(torch.long)
    num_confs = int((node_type_idx == 3).sum())
    paper_f = author_feats

    labels = np.load(osp.join(raw_dir, 'labels.npy'))
    labels = torch.from_numpy(labels).to(torch.long)

    s = {}
    N_a = author_feats.shape[0]
    N_p = paper_feats.shape[0]
    N_t = term_feats.shape[0]
    N_c = num_confs
    s['author'] = (0, N_a)
    s['paper'] = (N_a, N_a + N_p)
    s['term'] = (N_a + N_p, N_a + N_p + N_t)
    s['conference'] = (N_a + N_p + N_t, N_a + N_p + N_t + N_c)

    node_types = ['author', 'paper', 'term', 'conference']

    A = sparse.load_npz(osp.join(raw_dir, 'adjM.npz'))
    hg_data = dict()
    hg_data1 = dict()
    for src, dst in product(node_types, node_types):
        A_sub = A[s[src][0]:s[src][1], s[dst][0]:s[dst][1]]
        if A_sub.nnz > 0:
            hg_data[src, src + '_' + dst, dst] = A_sub.nonzero()


    AP = x = torch.sparse.FloatTensor(torch.LongTensor(hg_data[('author','author_paper','paper')]),
                             torch.FloatTensor(np.ones(hg_data[('author','author_paper','paper')][0].shape[0])),
                                      (author_feats.shape[0],paper_feats.shape[0]))

    PT = torch.sparse.FloatTensor(torch.LongTensor(hg_data[('paper', 'paper_term', 'term')]),
                             torch.FloatTensor(np.ones(hg_data[('paper', 'paper_term', 'term')][0].shape[0])),
                                      (paper_feats.shape[0],term_feats.shape[0]))
    PA = torch.sparse.FloatTensor(torch.LongTensor(hg_data[('paper','paper_author','author')]),
                             torch.FloatTensor(np.ones(hg_data[('paper','paper_author','author')][0].shape[0])),
                                      (paper_feats.shape[0],author_feats.shape[0]))
    PC = torch.sparse.FloatTensor(torch.LongTensor(hg_data[('paper','paper_conference','conference')]),
                             torch.FloatTensor(np.ones(hg_data[('paper','paper_conference','conference')][0].shape[0])),
                                      (paper_feats.shape[0],N_c))
    TP = torch.sparse.FloatTensor(torch.LongTensor(hg_data[('term', 'term_paper', 'paper')]),
                             torch.FloatTensor(np.ones(hg_data[('term', 'term_paper', 'paper')][0].shape[0])),
                                      (term_feats.shape[0],paper_feats.shape[0]))
    CP = torch.sparse.FloatTensor(torch.LongTensor(hg_data[('conference', 'conference_paper', 'paper')]),
                             torch.FloatTensor(np.ones(hg_data[('conference', 'conference_paper', 'paper')][0].shape[0])),
                                      (N_c,paper_feats.shape[0]))

    # APA,APCPA,APTPA
    APA = torch.matmul(AP,PA).indices().numpy()
    # APA = np.array_split(APA,2)
    APA = (APA[0], APA[1])

    APC = torch.matmul(AP,PC)
    CPA = torch.matmul(CP,PA)
    APCPA = torch.matmul(APC,CPA).indices().numpy()
    # APCPA = np.array_split(APCPA, 2)
    APCPA = (APCPA[0],APCPA[1])

    APT = torch.matmul(AP,PT)
    TPA = torch.matmul(TP,PA)
    APTPA = torch.matmul(APT,TPA).indices().numpy()
    # APTPA = np.array_split(APTPA, 2)
    APTPA = (APTPA[0], APTPA[1])

    # hg_data[]
    # APA
    hg_data1['a','a_p','p'] = (AP._indices().numpy()[0],AP._indices().numpy()[1])
    hg_data1['p','p_a','a'] = (PA._indices().numpy()[0],PA._indices().numpy()[1])
    # APCPA
    hg_data1['a','a_p_c','c'] = (APC.indices().numpy()[0],APC.indices().numpy()[1])
    hg_data1['c','c_p_a','a'] = (CPA.indices().numpy()[0],CPA.indices().numpy()[1])
    # APTPA
    hg_data1['a','a_p_t','t'] = (APT.indices().numpy()[0],APT.indices().numpy()[1])
    hg_data1['t','t_p_a','a'] = (TPA.indices().numpy()[0],TPA.indices().numpy()[1])
    # 
    # hg_data1["a_p","a_p_a","p_a"] = APA
    # hg_data1["a_p_c","a_p_c_p_a","c_p_a"] = APCPA
    # hg_data1["a_p_t","a_p_t_p_a","t_p_a"] = APTPA

    if feat == 1 or feat == 3 or feat == 4:
        pass
    elif feat == 2:
        hg_data['author', 'author_keyword', 'keyword'] = author_feats.nonzero()
        hg_data['keyword', 'keyword_author', 'author'] = author_feats.transpose().nonzero()
        hg_data['paper', 'paper_title', 'title-word'] = paper_feats.nonzero()
        hg_data['title-word', 'title_paper', 'paper'] = paper_feats.transpose().nonzero()
    hg = dgl.heterograph(hg_data)
    hg1 = dgl.heterograph(hg_data1)
    print(hg)
    print(hg1)
    split = np.load(osp.join(raw_dir, f'train_val_test_idx_{int(train_split * spllll)}_2021.npz'))
    # split = np.load(osp.join(raw_dir, f'train_val_test_idx.npz'))
    train_idx = split['train_idx']
    val_idx = split['val_idx']
    test_idx = split['test_idx']

    train_mask = get_binary_mask(len(labels), train_idx)
    val_mask = get_binary_mask(len(labels), val_idx)
    test_mask = get_binary_mask(len(labels), test_idx)

    num_classes = torch.unique(labels).shape[0]
    node_dict = {}
    edge_dict = {}
    for ntype in hg.ntypes:
        node_dict[ntype] = len(node_dict)
    for etype in hg.etypes:
        edge_dict[etype] = len(edge_dict)
        hg.edges[etype].data['id'] = torch.ones(hg.number_of_edges(etype), dtype=torch.long) * edge_dict[etype]

    ## 补充
    node_dict1 = {}
    edge_dict1 = {}
    for ntype in hg1.ntypes:
        node_dict1[ntype] = len(node_dict1)
    for etype in hg1.etypes:
        edge_dict1[etype] = len(edge_dict1)
        hg1.edges[etype].data['id'] = torch.ones(hg1.number_of_edges(etype),dtype=torch.long) * edge_dict1[etype]
    # return hg, node_dict, edge_dict, features, labels, num_classes, train_idx, val_idx, test_idx,\
    #        train_mask, val_mask, test_mask, target
    return hg,hg1, node_dict, edge_dict,paper_f, features, labels, num_classes, train_idx, val_idx, test_idx,\
           train_mask, val_mask, test_mask, target


def load_imdb_precessed(train_split=0.2, val_split=0.3, feat=1):
    raw_dir = './data/imdb/'

    node_types = ['movie', 'director', 'actor']
    movie_feats = sparse.load_npz(osp.join(raw_dir, 'features_0.npz'))   # 使用 .npz 格式从文件加载稀疏矩阵
    director_feats = sparse.load_npz(osp.join(raw_dir, 'features_1.npz'))
    actor_feats = sparse.load_npz(osp.join(raw_dir, 'features_2.npz'))
    target = 'movie'

    if feat == 1:
        movie_feats = torch.from_numpy(movie_feats.todense()).to(torch.float)
        director_feats = torch.from_numpy(director_feats.todense()).to(torch.float)
        actor_feats = torch.from_numpy(actor_feats.todense()).to(torch.float)
        features = {
            'movie': movie_feats,
            'director': director_feats,
            'actor': actor_feats
        }
    elif feat == 2 or feat == 4:
        features = {}
    elif feat == 3:
        movie_feats = torch.from_numpy(movie_feats.todense()).to(torch.float)
        features = {
            'movie': movie_feats
        }

    labels = np.load(osp.join(raw_dir, 'labels.npy'))  # 加载标签数据
    labels = torch.from_numpy(labels).to(torch.long)

    s = {}
    N_m = movie_feats.shape[0]  # 4278
    N_d = director_feats.shape[0]   # 2081
    N_a = actor_feats.shape[0]  # 5257
    s['movie'] = (0, N_m)
    s['director'] = (N_m, N_m + N_d)
    s['actor'] = (N_m + N_d, N_m + N_d + N_a)
    A = sparse.load_npz(osp.join(raw_dir, 'adjM.npz'))  # 邻接矩阵



    hg_data = dict()

    for src, dst in product(node_types, node_types):
        A_sub = A[s[src][0]:s[src][1], s[dst][0]:s[dst][1]]
        if A_sub.nnz > 0:
            hg_data[src, src + '_' + dst, dst] = A_sub.nonzero()

    if feat == 1 or feat == 3 or feat == 4:
        pass
    elif feat == 2:
        hg_data['movie', 'movie_keyword', 'keyword'] = movie_feats.nonzero()
        hg_data['keyword', 'keyword_movie', 'movie'] = movie_feats.transpose().nonzero()
    # 异构图
    hg = dgl.heterograph(hg_data)
    print(hg)

    split = np.load(osp.join(raw_dir, f'train_val_test_idx_{int(train_split * spllll)}_2021.npz'))
    # split = np.load(osp.join(raw_dir, f'train_val_test_idx.npz'))
    train_idx = split['train_idx']
    val_idx = split['val_idx']
    test_idx = split['test_idx']
    # 标签处理
    train_mask = get_binary_mask(len(labels), train_idx)
    val_mask = get_binary_mask(len(labels), val_idx)
    test_mask = get_binary_mask(len(labels), test_idx)

    num_classes = torch.unique(labels).shape[0] # 3
    node_dict = {}  # {'actor': 0, 'director': 1, 'movie': 2}
    edge_dict = {}  # {'actor_movie': 0, 'director_movie': 1, 'movie_actor': 2, 'movie_director': 3}
    for ntype in hg.ntypes:     # 提取节点类型
        node_dict[ntype] = len(node_dict)
    for etype in hg.etypes:     # 提取边类型
        edge_dict[etype] = len(edge_dict)
        hg.edges[etype].data['id'] = torch.ones(hg.number_of_edges(etype), dtype=torch.long) * edge_dict[etype]

    return hg,hg, node_dict, edge_dict,movie_feats, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, val_mask, test_mask, target


def load_data(dataset, train_split, val_split, feat=1):
    if dataset == 'acm':
        return load_acm_raw(train_split, val_split, feat=feat)
    elif dataset == 'dblp':
        return load_dblp_precessed(train_split, val_split, feat=feat)
    elif dataset == 'imdb':
        return load_imdb_precessed(train_split, val_split, feat=feat)
    else:
        return NotImplementedError('Unsupported dataset {}'.format(dataset))



def set_logger(my_str):
    task_time = get_date_postfix()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # Path(f"log/").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(f"log/{my_str}_{task_time}.log")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def get_checkpoint_path(args, epoch):
    Path('checkpoint').mkdir(parents=True, exist_ok=True)
    checkpoint_path = './checkpoint/{}_{}_{}_{}_{}_{}_{}_{}.pth'.format(
        args.prefix,
        args.dataset,
        args.feat,
        args.train_split,
        args.seed,
        args.n_hid,
        args.n_layers,
        epoch,
        args.max_lr)
    return checkpoint_path