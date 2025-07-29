import os.path as osp
import os
import shutil
import time
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch import nn
from torch.nn import functional as F
from model import SRHGN
from model import GAT
from utils import set_random_seed, load_data, get_n_params, set_logger
from metaview.utils import EarlyStopping,load_data_HAN
import seaborn as sns
from sklearn.decomposition import PCA
# sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", 10)

def load_params():
    parser = argparse.ArgumentParser(description='Training SR-HGN-HAN')
    parser.add_argument('--prefix', type=str, default='SR-HGN')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--feat', type=int, default=1)  #
    parser.add_argument('--seed', type=int, default=0)  # 随机种子
    parser.add_argument('--verbose', type=int, default=1)  #
    parser.add_argument('--train_split', type=float, default=0.2)  # 训练大小
    parser.add_argument('--val_split', type=float, default=0.3)  # 验证大小
    parser.add_argument('--num_type_heads', type=int, default=4)  # 类型数
    parser.add_argument('--num_node_heads', type=int, default=4)  # 节点头数
    parser.add_argument('--hidden_dim', type=int, default=256)# 隐藏维度
    parser.add_argument('--clip', type=int, default=1.0)  #
    parser.add_argument('--num_layers', type=int, default=3)  # 层
    parser.add_argument('--alpha', type=float, default=0.5)  # 参数

    parser.add_argument('--epochs', type=int, default=50)  # 训练轮数
    parser.add_argument('--dataset', type=str, default='acm')  # 数据集
    parser.add_argument('--input_dim', type=int, default=256)  # 输入维度
    parser.add_argument('--aggregate',type=str,default="concat",help="concat,mean,max")
    parser.add_argument('--iscuda', type=bool, default=True)  # 参数
    parser.add_argument('--cluster', action='store_true', default=True)

    args = parser.parse_args()
    args = vars(args)
    if args['aggregate'] == 'max':
        parser.add_argument('--max_lr', type=float, default=1e-3)  # 学习率
        parser.add_argument('--weight_decay', type=float, default=1e-5)  #
        parser.add_argument('--heads', type=int, default=8)  # 参数
        parser.add_argument('--han_heads', type=list, default=[4])  # 参数
        parser.add_argument('--hh_heads', type=list, default=3)  # 参数
        parser.add_argument('--dropout', type=float, default=0.2)
        args = parser.parse_args()
    elif args['aggregate'] == 'concat':
        parser.add_argument('--max_lr', type=float, default=1e-3)  # 学习率
        parser.add_argument('--weight_decay', type=float, default=1e-5)  #
        parser.add_argument('--heads', type=int, default=8)  # 参数
        parser.add_argument('--han_heads', type=list, default=[4])  # 参数
        parser.add_argument('--hh_heads', type=list, default=3)  # 参数
        parser.add_argument('--dropout', type=float, default=0.2)
        args = parser.parse_args()
    elif args['aggregate'] == 'mean':
        parser.add_argument('--max_lr', type=float, default=1e-3)  # 学习率
        parser.add_argument('--weight_decay', type=float, default=1e-5)  #
        parser.add_argument('--heads', type=int, default=8)  # 参数
        parser.add_argument('--han_heads', type=list, default=[4])  # 参数
        parser.add_argument('--hh_heads', type=list, default=3)  # 参数
        parser.add_argument('--dropout', type=float, default=0.2)
        args = parser.parse_args()

    return vars(args)


def init_feat(G, n_inp, features):
    # Randomly initialize features if features don't exist
    input_dims = {}

    for ntype in G.ntypes:
        emb = nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), n_inp), requires_grad=True)
        nn.init.xavier_uniform_(emb)  # 初始化矩阵

        feats = features.get(ntype, emb)
        G.nodes[ntype].data['x'] = feats  # 保存特征值
        input_dims[ntype] = feats.shape[1]

    return G, input_dims


def train(model, G, G_HAN, labels, target, optimizer, scheduler, train_idx, feature, clip=1.0):
    model.train()
    # G = G.to("cuda:0")
    logits, _, _ = model(G,G_HAN, target,feature)
    # logits = model(G, target)
    loss = F.cross_entropy(logits[train_idx], labels[train_idx])

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    scheduler.step()

    return loss.item()


def eval(model, G, G_HAN, labels, target, train_idx, val_idx, test_idx,feature):
    model.eval()

    logits, _, _ = model(G,G_HAN, target,feature)
    # logits = model(G, target)
    pred = logits.argmax(1).detach().cpu().numpy()

    train_macro_f1 = f1_score(labels[train_idx].cpu(), pred[train_idx], average='macro')
    train_micro_f1 = f1_score(labels[train_idx].cpu(), pred[train_idx], average='micro')
    val_macro_f1 = f1_score(labels[val_idx].cpu(), pred[val_idx], average='macro')
    val_micro_f1 = f1_score(labels[val_idx].cpu(), pred[val_idx], average='micro')
    test_macro_f1 = f1_score(labels[test_idx].cpu(), pred[test_idx], average='macro')
    test_micro_f1 = f1_score(labels[test_idx].cpu(), pred[test_idx], average='micro')

    return {
        'train_maf1': train_macro_f1,
        'train_mif1': train_micro_f1,
        'val_maf1': val_macro_f1,
        'val_mif1': val_micro_f1,
        'test_maf1': test_macro_f1,
        'test_mif1': test_micro_f1
    }


def cluster(model, G,G_HAN, target, labels,feature,epoch,dataset,time):
    model.eval()
    
    _, embedding, attns = model(G,G_HAN, target,feature)
    # huitu(embedding,labels,epoch,dataset,time)
    embedding = embedding.detach().cpu().numpy()
    labels = labels.cpu()

    kmeans = KMeans(n_clusters=len(torch.unique(labels)), random_state=42).fit(embedding)
    nmi = normalized_mutual_info_score(labels, kmeans.labels_)
    ari = adjusted_rand_score(labels, kmeans.labels_)

    return {
        'nmi': nmi,
        'ari': ari
    }

def huitu(data,label,epoch,dataset,time):
    data = data.detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    X_pca= PCA(n_components=32).fit_transform(data)
    X_tsne = TSNE(n_components=2,random_state=33).fit_transform(X_pca)

    # X_tsne = TSNE(n_components=2,init='pca',random_state=33).fit_transform(data)

    ckpt_dir = "images"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    plt.figure(figsize=(8,8))
    ax = plt.subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.scatter(X_tsne[:,0],X_tsne[:,1],c=label)
    # plt.legend()
    # plt.subplot(122)
    # plt.scatter(X_pca[:,0],X_pca[:,1],c=label)
    # plt.legend()
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.savefig(f'images/digit_{dataset}_{int(time)}_{epoch}.png',dpi=120)
    plt.show()


def main(params):
    if params['iscuda']:
        device = torch.device(f"cuda:0" if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    my_str = f"{params['prefix']}_{params['dataset']}"

    logger = set_logger(my_str)
    logger.info(params)

    checkpoints_path = f'checkpoints'
    Path(checkpoints_path).mkdir(parents=True, exist_ok=True)

    G, G_HAN, node_dict, edge_dict,feature, features, labels, num_classes, \
    train_idx, val_idx, test_idx, train_mask, val_mask, test_mask, target = load_data(
        params['dataset'], params['train_split'], params['val_split'], params['feat'])

    # {'actor': 3066, 'director': 3066, 'movie': 3066}
    G, input_dims = init_feat(G, params['input_dim'], features)
    G = G.to(device)
    G_HAN = G_HAN.to(device)
    labels = labels.to(device)
    feature = feature.to(device)

    # model = SRHGN(G,
    #               node_dict, edge_dict,
    #               input_dims=input_dims,
    #               hidden_dim=params['hidden_dim'],  # 256
    #               output_dim=labels.max().item() + 1,   # 3
    #               num_layers=params['num_layers'],  # 3
    #               num_node_heads=params['num_node_heads'],  # 4
    #               num_type_heads=params['num_type_heads'],  # 4
    #               alpha=params['alpha']).to(device)
    print((params['dataset']))
    if params['dataset'] == 'acm':
        model = GAT(G,
                    node_dict, edge_dict,
                    input_dims=input_dims,
                    hidden_dim=params['hidden_dim'],   # 256
                    output_dim=labels.max().item() + 1,
                    num_layers=params['num_layers'],
                    in_size=feature.shape[1],
                    nums_head=params['han_heads'],
                    num_node_heads=params['num_node_heads'],
                    num_type_heads=params['num_type_heads'],
                    alpha=params['alpha'],
                    heads=params['heads'],
                    dropout=params['dropout'],
                    aggregate=params['aggregate'],
                    hh_heads=params['hh_heads'],
                    iscuda=params['iscuda'],
                    meta_paths=[
                        # ["paper_paper_cite", "paper_paper_ref"],
                        # ["paper_term","term_paper"],
                        ["paper_author", "author_paper"],
                        ["paper_subject", "subject_paper"]],
                    ).to(device)
    elif params['dataset'] == 'dblp':
        model = GAT(G,
                    node_dict, edge_dict,
                    input_dims=input_dims,
                    hidden_dim=params['hidden_dim'],   # 256
                    output_dim=labels.max().item() + 1,
                    num_layers=params['num_layers'],
                    in_size=feature.shape[1],
                    nums_head=params['han_heads'],
                    num_node_heads=params['num_node_heads'],
                    num_type_heads=params['num_type_heads'],
                    alpha=params['alpha'],
                    heads=params['heads'],
                    dropout=params['dropout'],
                    aggregate=params['aggregate'],
                    hh_heads=params['hh_heads'],
                    iscuda=params['iscuda'],
                    meta_paths=[
                        ['a_p', 'p_a'],
                        ['a_p_c', 'c_p_a'],
                        ['a_p_t', 't_p_a']],
                    ).to(device)
    elif params['dataset'] == 'imdb':
        model = GAT(G,
                    node_dict, edge_dict,
                    input_dims=input_dims,
                    hidden_dim=params['hidden_dim'],  # 256
                    output_dim=labels.max().item() + 1,
                    num_layers=params['num_layers'],
                    in_size=feature.shape[1],
                    nums_head=params['han_heads'],
                    num_node_heads=params['num_node_heads'],
                    num_type_heads=params['num_type_heads'],
                    alpha=params['alpha'],
                    heads=params['heads'],
                    dropout=params['dropout'],
                    aggregate=params['aggregate'],
                    hh_heads=params['hh_heads'],
                    iscuda=params['iscuda'],
                    meta_paths=[
                        ['movie_director', 'director_movie'],
                        ['movie_actor', 'actor_movie']]
                    ).to(device)
    """
    ["paper_paper_cite","paper_paper_ref"],
    # ["paper_term","term_paper"],
    ["paper_author", "author_paper"],
    ["paper_subject", "subject_paper"]],
    
    ['author_paper','paper_author'],
    ['term_paper','paper_term'],
    ['paper_conference','conference_paper']
                        
    """

    t = time.localtime()
    tt = time.time()

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=params['weight_decay'])  # L2正则化，权重衰退
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=params['epochs'], max_lr=params['max_lr'])
    logger.info('Training SR-HGN with #param: {:d}'.format(get_n_params(model)))

    best_test_maf1 = 0
    best_epoch = 0
    print("----------------------"+params['aggregate']+" pooling-------------------------")

    for epoch in range(1, params['epochs'] + 1):
        loss = train(model, G, G_HAN, labels, target, optimizer, scheduler, train_idx, feature, clip=params['clip'])

        if epoch % params['verbose'] == 0:
            results = eval(model, G,G_HAN, labels, target, train_idx, val_idx, test_idx, feature)

            if results['test_maf1'] >= best_test_maf1:
                best_test_maf1 = results['test_maf1']
                best_results = results
                best_epoch = epoch

                torch.save(model.state_dict(), osp.join(checkpoints_path, f'{my_str}_{epoch}.pkl'))
                
                if params['cluster']:
                    model.load_state_dict(torch.load(osp.join(checkpoints_path, f'{my_str}_{best_epoch}.pkl')))
                    cluster_results = cluster(model, G, G_HAN, target, labels, feature,epoch,dataset=params['dataset'],time=tt)

                    logger.info('ARI: {:.4f} | NMI: {:.4f}'.format(cluster_results['ari'], cluster_results['nmi']))
            # 'Val MaF1: {:.4f} (Best: {:.4f}) | MiF1: {:.4f} (Best: {:.4f}) |\n'
            logger.info(
                'Epoch: {:d} | LR: {:.4f} | Loss {:.4f} | '

                'Val MaF1: {:.4f} (Best: {:.4f}) | '
                'Test MaF1: {:.4f} (Best: {:.4f}) | MiF1: {:.4f} (Best: {:.4f})'.format(
                    epoch, optimizer.param_groups[0]['lr'], loss,
                    # results['train_maf1'], best_results['train_maf1'],
                    # results['train_mif1'], best_results['train_mif1'],
                    results['val_maf1'], best_results['val_maf1'],
                    # results['val_mif1'],best_results['val_mif1'],
                    results['test_maf1'], best_results['test_maf1'],
                    results['test_mif1'], best_results['test_mif1']

                ))
            
            # if(epoch==25):
            #     huitu(feature,labels,epoch)

            # torch.save(model.state_dict(), osp.join(checkpoints_path, f'{my_str}_{epoch}.pkl'))

    logger.info(
        'Best Epoch: {:d} | Train MiF1: {:.4f},  MaF1: {:.4f} | Val MiF1: {:.4f}, MaF1: {:.4f} | Test MaF1: {:.4f}, MiF1: {:.4f}'.format(
            best_epoch,
            best_results['train_mif1'],
            best_results['train_maf1'],
            best_results['val_mif1'],
            best_results['val_maf1'],
            best_results['test_maf1'],
            best_results['test_mif1'],

        ))

    if params['cluster']:
        model.load_state_dict(torch.load(osp.join(checkpoints_path, f'{my_str}_{best_epoch}.pkl')))
        cluster_results = cluster(model, G, G_HAN, target, labels,feature,best_epoch,dataset=params['dataset'],time=tt)

        logger.info('ARI: {:.4f} | NMI: {:.4f}'.format(cluster_results['ari'],cluster_results['nmi']))

# 聚类


if __name__ == '__main__':
    params = load_params()
    set_random_seed(params['seed'])
    main(params)
