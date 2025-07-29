import dgl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.functional import edge_softmax
from dgl.nn.pytorch import GATConv
import numpy as np

aggregate = 'max'
class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)

        return (beta * z).sum(1)  # (N, D * K)


class HANLayer(nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    meta_paths : list of metapaths, each as a list of edge types
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    g : DGLGraph
        The heterogeneous graph
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """

    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.gat_layers.append(
                GATConv(
                    in_size,
                    out_size,
                    layer_num_heads,
                    dropout,
                    dropout,
                    activation=F.elu,
                    allow_zero_in_degree=True,
                )  # 4*256
            )
        self.semantic_attention = SemanticAttention(
            in_size=out_size * layer_num_heads
        )
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        semantic_embeddings = []

        # semantic_embeddings = np.array(semantic_embeddings)  # list转numpy.array
        # semantic_embeddings = torch.from_numpy(semantic_embeddings)  # array2tensor

        # print("no problem")
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[
                    meta_path
                ] = dgl.metapath_reachable_graph(g, meta_path)
        # print("no problem")
        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            temp = self.gat_layers[i](new_g, h).flatten(1)
            if aggregate=='max':
                if i==0:
                    x1 = temp
                else:
                    x1 = torch.stack((x1,temp),dim=0)
                    x1 = torch.max(x1,dim=0).values
            if aggregate=='mean':
                if i==0:
                    x1 = temp
                else:
                    
                    x1 = torch.add(x1,temp)
            
            
            semantic_embeddings.append(temp)
        semantic_embeddings = torch.stack(
            semantic_embeddings, dim=1
        )  # (N, M, D * K)
        # x1 = x1/3.0
        # print("no problem")
        # return self.semantic_attention(semantic_embeddings)  # (N, D * K)
        if aggregate=='max':
            return x1  # (N, D * K)
        else:
            return x1


class HAN(nn.Module):
    def __init__(
            self, meta_paths, in_size, hidden_size, out_size, num_heads, dropout
    ):
        super(HAN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(
            HANLayer(meta_paths, in_size, hidden_size, num_heads[0], dropout)
        )
        for l in range(1, len(num_heads)):
            self.layers.append(
                HANLayer(
                    meta_paths,
                    hidden_size * num_heads[l - 1],
                    hidden_size,
                    num_heads[l],
                    dropout,
                )
            )
        # self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)

        # return self.predict(h)
        return h


class SRHGNLayer(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 node_dict,
                 edge_dict,
                 num_node_heads=4,
                 num_type_heads=4,
                 dropout=0.2,
                 alpha=0.5,
                 ):
        super(SRHGNLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.num_types = len(node_dict)  # 
        self.num_relations = len(edge_dict)  # 

        self.total_rel = self.num_types * self.num_relations * self.num_types
        self.num_node_heads = num_node_heads
        self.num_type_heads = num_type_heads

        self.node_linear = nn.ModuleList()  # 
        self.edge_linear = nn.ModuleList()  # 

        self.src_attn = nn.ModuleList()
        self.dst_attn = nn.ModuleList()

        self.sem_attn_src = nn.ModuleList()
        self.sem_attn_dst = nn.ModuleList()
        self.rel_attn = nn.ModuleList()

        for _ in range(self.num_types):  # 
            self.node_linear.append(nn.Linear(input_dim, output_dim))

        for _ in range(self.num_relations):  # 
            self.edge_linear.append(nn.Linear(input_dim, output_dim))
            self.src_attn.append(nn.Linear(input_dim, num_node_heads))
            self.dst_attn.append(nn.Linear(input_dim, num_node_heads))

            self.sem_attn_src.append(nn.Linear(output_dim, num_type_heads))
            self.sem_attn_dst.append(nn.Linear(output_dim, num_type_heads))
            self.rel_attn.append(nn.Linear(output_dim, num_type_heads))

        # Assign learnable relation embedding
        self.rel_emb = nn.Parameter(torch.randn(self.num_relations, output_dim), requires_grad=True)  # 
        nn.init.xavier_normal_(self.rel_emb, gain=1.414)

        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float), requires_grad=False)
        self.epsilon = nn.Parameter(torch.FloatTensor([1e-9]), requires_grad=False)  # 

        self.drop = nn.Dropout(dropout)

    def forward(self, G, h):
        with G.local_scope():

            node_dict, edge_dict = self.node_dict, self.edge_dict

            for src, e, dst in G.canonical_etypes:  # 
                # Extract subgraph for each relation  
                sub_graph = G[src, e, dst]
                h_src = h[src]  # 
                h_dst = h[dst]  # 

                e_id = edge_dict[e]  # 
                src_id = node_dict[src]  # 
                dst_id = node_dict[dst]  # 

                # 
                h_src = self.drop(self.edge_linear[e_id](h_src))

                # 
                h_dst = self.drop(self.node_linear[dst_id](h_dst))

                # Calculate attention score similar to GAT
                src_attn = self.drop(self.src_attn[src_id](h_src)).unsqueeze(-1)
                dst_attn = self.drop(self.dst_attn[dst_id](h_dst)).unsqueeze(-1)

                # 
                sub_graph.srcdata.update({'attn_src': src_attn})
                sub_graph.dstdata.update({'attn_dst': dst_attn})
                sub_graph.apply_edges(fn.u_add_v('attn_src', 'attn_dst', 'a'))
                a = F.leaky_relu(sub_graph.edata['a'])

                # 
                sub_graph.srcdata[f'v_{e_id}'] = h_src.view(
                    -1, self.num_node_heads, self.output_dim // self.num_node_heads)  # Multi-head attention
                sub_graph.edata[f'a_{e_id}'] = self.drop(edge_softmax(sub_graph, a))

            # Aggregate type-level embedding like GAT
            # z: # nodes x # relations x # heads x dim [N x R x H x (D // H)]
            G.multi_update_all({etype: (fn.u_mul_e(f'v_{e_id}', f'a_{e_id}', 'm'), fn.sum('m', 'z')) \
                                for etype, e_id in edge_dict.items()}, cross_reducer='stack')

            z = {}
            attns = {}
            rel_idx_start = 0

            for ntype in G.ntypes:
                dst_id = node_dict[ntype]
                h_dst = h[ntype]

                z_src = G.nodes[ntype].data['z']  # [N x R x H x (D // H)]
                num_nodes = z_src.shape[0]
                num_rel = z_src.shape[1]

                z_src = z_src.view(num_nodes, num_rel, self.output_dim)  # [N x R x D]
                z_dst = self.drop(self.node_linear[dst_id](h_dst))  # [N x D]

                sem_attn = torch.zeros(num_nodes, num_rel, self.num_type_heads, device=z_src.device)  # [N x R x H]
                rel_attn = torch.zeros(num_nodes, num_rel, self.num_type_heads, device=z_src.device)  # [N x R x H]

                # Compute semantic-aware and relation-aware attention scores
                for rel_idx in range(num_rel):
                    normalize = lambda x: x / (torch.max(torch.norm(x, dim=1, keepdim=True), self.epsilon))

                    attn_idx = rel_idx_start + rel_idx
                    z_src_rel = z_src[:, rel_idx]

                    sem_attn_src = self.sem_attn_src[attn_idx](normalize(z_src_rel))
                    sem_attn_dst = self.sem_attn_dst[attn_idx](normalize(z_dst))

                    sem_attn[:, rel_idx] = sem_attn_src + sem_attn_dst
                    rel_attn[:, rel_idx] = self.rel_attn[attn_idx](self.rel_emb[attn_idx].unsqueeze(0)).repeat(
                        num_nodes, 1)

                rel_idx_start += num_rel

                sem_attn = self.drop(F.softmax(F.leaky_relu(sem_attn), dim=1))
                rel_attn = self.drop(F.softmax(F.leaky_relu(rel_attn), dim=1))
                # 
                attn = self.alpha * sem_attn + (1 - self.alpha) * rel_attn

                # if aggregate=='max':
                #     zz = z_src.view(num_nodes, num_rel, self.num_type_heads, -1)
                #     zz = torch.max(zz,dim=-2)
                #     zz = F.gelu(zz+h[ntype])
                #     z[ntype] = normalize(zz)
                #     attns[ntype] = {'full': attn.detach().cpu().numpy(),
                #                 'semantic': sem_attn.detach().cpu().numpy(),
                #                 'relation': rel_attn.detach().cpu().numpy()}

                # Multiple multi-head attention and node embedding
                # z_dst = torch.mul(z_src.view(num_nodes, num_rel, self.num_type_heads, -1),
                                #   attn.unsqueeze(-1))  # [N x R x H x (D // H)]
                z_dst = z_src.view(num_nodes,num_rel,self.num_type_heads,-1)
                # Concatenate all heads
                z_dst = z_dst.view(num_nodes, num_rel, self.output_dim)  # [N x R x D]

                # 
                z_dst = F.gelu(z_dst.sum(1) + h[ntype])

                z[ntype] = normalize(z_dst)

                attns[ntype] = {'full': attn.detach().cpu().numpy(),
                                'semantic': sem_attn.detach().cpu().numpy(),
                                'relation': rel_attn.detach().cpu().numpy()}
            return z, attns


class SRHGN(nn.Module):
    def __init__(self,
                 G,
                 node_dict,
                 edge_dict,
                 input_dims,
                 hidden_dim,
                 output_dim,
                 num_layers=2,
                 num_node_heads=4,
                 num_type_heads=4,
                 alpha=0.5
                 ):
        super(SRHGN, self).__init__()

        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        # 
        self.pre_transform = nn.ModuleList()
        for ntype, idx in node_dict.items():
            self.pre_transform.append(nn.Linear(input_dims[ntype], hidden_dim))  #   # 
        # 
        self.convs = nn.ModuleList()
        for _ in range(num_layers):  # 3
            self.convs.append(
                SRHGNLayer(
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    node_dict=node_dict,
                    edge_dict=edge_dict,
                    num_node_heads=num_node_heads,
                    num_type_heads=num_type_heads,
                    alpha=alpha
                ))

        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, G, target):
        h = {}  # 
        attns = []

        # Pre-transformation
        for ntype in G.ntypes:
            n_id = self.node_dict[ntype]
            h[ntype] = self.pre_transform[n_id](G.nodes[ntype].data['x'])
            h[ntype] = F.gelu(h[ntype])  # 
            # h[ntype] = F.leaky_relu(h[ntype],negative_slope=1e-2)
            # h[ntype] = F.tanh(h[ntype])

        for conv in self.convs:
            h, attn = conv(G, h)
            attns.append(attn)

        # logits = self.out(h[target])

        # return logits, h[target], attns
        return h[target], attns


class GAT(nn.Module):
    def __init__(self,
                 G,
                 node_dict,
                 edge_dict,
                 input_dims,
                 hidden_dim,
                 output_dim,
                 meta_paths,
                 in_size,
                 nums_head,
                 hh_heads,
                 aggregate,
                 num_layers=2,
                 num_node_heads=4,
                 num_type_heads=4,
                 alpha=0.5,
                 heads=6,
                 dropout=0.2,
                 iscuda=False
                 ):
        super(GAT, self).__init__()
        # self.dropout = dropout
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.nheads = heads
        self.dropout = dropout
        self.aggregate = aggregate
        self.hh_heads = hh_heads

        self.parameter1 = nn.Parameter(torch.tensor([0.5]))
        self.parameter2 = nn.Parameter(torch.tensor([0.5]))
        # self.parameter1 = nn.Parameter(torch.randn(1))
        # self.parameter2 = nn.Parameter(torch.randn(1))

        self.attentions = [SRHGN(G=G,
                                 node_dict=self.node_dict,
                                 edge_dict=self.edge_dict,
                                 input_dims=self.input_dims,
                                 hidden_dim=self.hidden_dim,  # 256
                                 output_dim=self.output_dim,  # 3
                                 num_layers=self.num_layers,
                                 num_node_heads=num_node_heads,
                                 num_type_heads=num_type_heads,
                                 alpha=alpha) for _ in range(self.nheads)]
        # self.attentions2 = [HAN(meta_paths=meta_paths,
        #                         in_size=in_size,  # 1902
        #                         hidden_size=hidden_dim,  # 256
        #                         out_size=output_dim,  # 3
        #                         num_heads=nums_head,
        #                         dropout=self.dropout) for _ in range(self.nheads)]
        if iscuda:
            self.attentions2 = [HAN(meta_paths=meta_paths,
                                    in_size=in_size,  # 1902
                                    hidden_size=hidden_dim,  # 256
                                    out_size=output_dim,  # 3
                                    num_heads=nums_head,
                                    dropout=self.dropout) for _ in  range(self.hh_heads)]
        else:
            self.attentions2 = [HAN(meta_paths=meta_paths,
                                    in_size=in_size,  # 1902
                                    hidden_size=hidden_dim,  # 256
                                    out_size=output_dim,  # 3
                                    num_heads=nums_head,
                                    dropout=self.dropout).cuda() for _ in range(self.hh_heads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        for i, attention in enumerate(self.attentions2):
            self.add_module('attention2_{}'.format(i), attention)


        if self.aggregate == 'mean':
            """"""
            self.out_mean_x1 = nn.Linear(self.hidden_dim, self.output_dim)
            self.out_mean_x2 = nn.Linear(self.hidden_dim * nums_head[0], self.output_dim)
        elif self.aggregate == 'concat':
            """"""
            self.out_concat_x1 = nn.Linear(self.hidden_dim * self.nheads, self.output_dim)
            self.out_concat_x1_64 = nn.Linear(self.hidden_dim * self.nheads, 32)

            self.out_concat_x2_1 = nn.Linear(self.hidden_dim * self.hh_heads * nums_head[0],
                                             self.hidden_dim * nums_head[0])
            
            self.out_concat_x2_2 = nn.Linear(self.hidden_dim * nums_head[0], self.output_dim)
            self.out_concat_x2_2_64 = nn.Linear(self.hidden_dim * nums_head[0], 32)
        else:
            self.out_max_x1 = nn.Linear(self.hidden_dim, self.output_dim)
            self.out_max_x2 = nn.Linear(self.hidden_dim * nums_head[0], self.output_dim)

        self.out_total = nn.Linear(self.hidden_dim * self.nheads + self.hidden_dim * nums_head[0], self.output_dim)

        # self.out2 = nn.Linear(self.hidden_dim*self.nheads*nums_head[0],self.hidden_dim*self.nheads)

    def forward(self, G, G_HAN, target, features):

        # x1 = torch.cat(x1,dim=1)
        # x2 = torch.cat(x2,dim=1)
        # x2 = F.elu(self.out_x2(x2))

        sigmoid_parameter1 = torch.sigmoid(self.parameter1)
        # sigmoid_parameter2 = torch.tensor([0.2]).cuda()
        sigmoid_parameter2 = torch.sigmoid(self.parameter2)

        ssss = torch.tensor([0.2])

        # numb = torch.tensor()

        if self.aggregate == 'mean':
            """"""
            # print("{}   {}".format(sigmoid_parameter1, 1 - sigmoid_parameter1))
            for i, att in enumerate(self.attentions):
                h1, attns = att(G, target)
                if i == 0:
                    x1 = h1
                else:
                    x1 = torch.add(x1, h1)
            for i,att in enumerate(self.attentions2):
                h2 = att(G_HAN, features)
                if i == 0:
                    x2 = h2
                else:
                    x2 = torch.add(x2, h2)

            x1 = F.dropout(x1/self.nheads, self.dropout, training=self.training)
            x1 = F.tanh(self.out_mean_x1(x1))
            x2 = F.dropout(x2/self.hh_heads, self.dropout, training=self.training)
            x2 = F.tanh(self.out_mean_x2(x2))
            xlist = F.tanh(torch.add(x1 * sigmoid_parameter2, x2 * (1 - sigmoid_parameter2)))
            # print("{}   {}".format(sigmoid_parameter2, 1 - sigmoid_parameter2))
            return F.log_softmax(xlist, dim=1), xlist, attns

        elif self.aggregate == 'concat':
            """"""
            x1 = []
            x2 = []
            for att in self.attentions:
                h1, attns = att(G, target)
                x1.append(h1)
            for att in self.attentions2:
                h2 = att(G_HAN, features)
                x2.append(h2)
            x1 = torch.cat(x1, dim=1)
            x2 = torch.cat(x2, dim=1)
            x1 = F.dropout(x1, self.dropout, training=self.training)
            
            # x3 = F.tanh(self.out_concat_x1_64(x1))
            # x4 = F.tanh(self.out_concat_x2_1(x2))
            # x4 = F.tanh(self.out_concat_x2_2_64(x4))


            x1 = F.tanh(self.out_concat_x1(x1))

            x2 = F.dropout(x2, self.dropout, training=self.training)
            x2 = F.tanh(self.out_concat_x2_1(x2))
            x2 = F.tanh(self.out_concat_x2_2(x2))
            # xlist = F.elu(torch.add(x1, x2))
            # xlist = F.elu(torch.add(x1 * sigmoid_parameter2, x2 * sigmoid_parameter1))
            # x4 = F.elu(torch.add(x3*sigmoid_parameter2,x4*(1-sigmoid_parameter2)))
            xlist = F.tanh(torch.add(x1 * sigmoid_parameter2, x2 * (1-sigmoid_parameter2)))
            # print("{}   {}".format(sigmoid_parameter1, sigmoid_parameter2))
            # print("{}   {}".format(sigmoid_parameter2, 1-sigmoid_parameter2))
            return F.log_softmax(xlist, dim=1), xlist, attns

        elif self.aggregate == 'max':
            """max"""
            # x1 = torch.
            for i, att in enumerate(self.attentions):
                h1, attns = att(G, target)
                if i == 0:
                    x1 = h1
                else:
                    x1 = torch.stack((x1, h1), dim=0)
                    x1 = torch.max(x1, dim=0).values

            for i, att in enumerate(self.attentions2):
                h2 = att(G_HAN, features)
                if i == 0:
                    x2 = h2
                else:
                    x2 = torch.stack((x2, h2), dim=0)
                    x2 = torch.max(x2, dim=0).values

            x1 = F.dropout(x1, self.dropout, training=self.training)
            x1 = F.tanh(self.out_max_x1(x1))
            x2 = F.tanh(self.out_max_x2(x2))
            xlist = F.tanh(torch.add(x1 * sigmoid_parameter2, x2 * (1 - sigmoid_parameter2)))
            print("{}   {}".format(sigmoid_parameter2, 1 - sigmoid_parameter2))

            return F.log_softmax(xlist, dim=1), xlist, attns

        # print("{:.4f}  {:.4f} ".format(1-sigmoid_parameter,sigmoid_parameter))
        # print("{:.4f}  {:.4f}".format(sigmoid_parameter1,sigmoid_parameter2))






