import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter



class HypergraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels ):
        super(HypergraphConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.W = nn.Parameter(torch.Tensor(in_channels, out_channels).to(torch.bfloat16))
        self.U = nn.Parameter(torch.Tensor(out_channels, out_channels).to(torch.bfloat16))
        self.reset_parameters()

    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.U)
    
    def forward(self, x, H):
        '''
        :param: x: BLFN   bffloat16
        :param: H: hypergraph matrix shape (n,m)  bffloat16
        '''
        device = self.W.device  # avoid calculating at two device 
        x = x.to(device)
        H = H.to(device)
        # Node to hyperedge information aggregation 
        x_enc = torch.matmul(x, self.W)  # (B, L, N, F) ->(B, L, N, F)
        x_enc = torch.einsum('nm, blnf->blmf', H, x_enc)  # (N, M), (B, L, N, F') -> (B, L, M, F')
        # information transformation on hyperedges
        x_enc = F.relu(torch.matmul(x_enc,self.U))  # (B, L, M, F') -> (B, L, M, F')
        # Hyperedges to Node information aggregation
        x_out = torch.einsum('mn,blmf->blnf',  H.t(), x_enc) #  (N, M), (B, L, M, F') -> (B, L, N, F')
        return x_out



class HypergraphSAGE(nn.Module):
    def __init__(self, in_channels, out_channels,dropout=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Initial Parameters
        self.W_self = nn.Parameter(torch.Tensor(in_channels, out_channels).to(torch.bfloat16))
        self.W_neigh = nn.Parameter(torch.Tensor(in_channels, out_channels).to(torch.bfloat16))
        self.W_concat = nn.Parameter(torch.Tensor(2 * out_channels, out_channels).to(torch.bfloat16))
        self.reset_parameters()


    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_self)
        nn.init.xavier_uniform_(self.W_neigh)
        nn.init.xavier_uniform_(self.W_concat)        


    def forward(self, x, H):
        '''
        :param x: node feature matrix, shape (B, L, N, F)
        :param H: hypergraph Hacency matrix, shape (N, M) (N: number of nodes, M: number of hyperedges)
        :return: indicates the node feature after aggregation
        '''
        device = self.W_self.device
        x = x.to(device)
        H = H.to(device)

        # node degree and hyperedges degree
        hyperedge_degrees = H.sum(dim=0).to(torch.bfloat16)  # (M,)
        node_degrees = H.sum(dim=1).to(torch.bfloat16)       # (N,)

        
        # Node feature transformation
        x_self = torch.einsum('blnf, fd -> blnd', x, self.W_self)  # (B, L, N, F')

        # hyperedges feature aggregation: Node features in aggregate hyperedges (averaging pooling)
        hyperedge_features = torch.einsum('mn, blnf -> blmf', H.t(), x)  # (B, L, M, F)
        hyperedge_features = hyperedge_features / (hyperedge_degrees + 1).unsqueeze(-1).unsqueeze(0).unsqueeze(0)
        hyperedge_features = self.relu(hyperedge_features)

        # Aggregate hyperedges features to nodes
        x_neigh = torch.einsum('nm, blmf -> blnf', H, hyperedge_features)  # (B, L, N, F)
        x_neigh = x_neigh / (node_degrees + 1).unsqueeze(-1).unsqueeze(0).unsqueeze(0)
        x_neigh = torch.einsum('blnf, fd -> blnd', x_neigh, self.W_neigh)  # (B, L, N, F')
        
        # ==== 4. 特征拼接与融合 ====
        x_concat = torch.cat([x_self, x_neigh], dim=-1)  # (B, L, N, 2F')
        x_out = torch.einsum('blnd, df -> blnf', x_concat, self.W_concat)  # (B, L, N, F'')
        
        return x_out        




class HypergraphAttention(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super(HypergraphAttention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout

        # 初始化参数
        self.W_self = nn.Parameter(torch.Tensor(in_channels, out_channels).to(torch.bfloat16))
        self.W_neigh = nn.Parameter(torch.Tensor(in_channels, out_channels).to(torch.bfloat16))
        self.W_concat = nn.Parameter(torch.Tensor(2 * out_channels, out_channels).to(torch.bfloat16))
        self.attention = nn.Parameter(torch.Tensor(2 * out_channels, 1).to(torch.bfloat16))

        # 初始化权重
        nn.init.xavier_uniform_(self.W_self)
        nn.init.xavier_uniform_(self.W_neigh)
        nn.init.xavier_uniform_(self.W_concat)
        nn.init.xavier_uniform_(self.attention)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, H):
        """
        :param x: 节点特征矩阵，形状 (B, L, N, F)
        :param H: 超图邻接矩阵，形状 (N, M) (N: 节点数, M: 超边数)
        :return: 聚合后的节点特征
        """
        device = self.W_self.device
        x = x.to(device)
        H = H.to(device)

        # 节点特征转换
        x_self = torch.einsum('blnf, fd -> blnd', x, self.W_self)  # (B, L, N, F')

        # 超边特征聚合
        hyperedge_features = torch.einsum('mn, blnf -> blmf', H.t(), x)  # (B, L, M, F)
        hyperedge_features = hyperedge_features / (H.sum(dim=0) + 1).unsqueeze(-1).unsqueeze(0).unsqueeze(0)
        hyperedge_features = self.leaky_relu(hyperedge_features)

        # 聚合超边特征到节点
        x_neigh = torch.einsum('nm, blmf -> blnf', H, hyperedge_features)  # (B, L, N, F)
        x_neigh = x_neigh / (H.sum(dim=1) + 1).unsqueeze(-1).unsqueeze(0).unsqueeze(0)
        x_neigh = torch.einsum('blnf, fd -> blnd', x_neigh, self.W_neigh)  # (B, L, N, F')

        # 特征拼接
        x_concat = torch.cat([x_self, x_neigh], dim=-1)  # (B, L, N, 2F')

        # 计算注意力权重
        attention_scores = torch.einsum('blnd, df -> bln', x_concat, self.attention)  # (B, L, N)
        attention_weights = self.softmax(attention_scores)  # (B, L, N)
        attention_weights = self.dropout_layer(attention_weights)

        # 应用注意力权重
        x_attention = x_concat * attention_weights.unsqueeze(-1)  # (B, L, N, 2F')

        # 最终特征融合
        x_out = torch.einsum('blnd, df -> blnf', x_attention, self.W_concat)  # (B, L, N, F'')

        return x_out
    
