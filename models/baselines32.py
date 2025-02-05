from math import sqrt

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer

from layers.Embed import PatchEmbedding
from layers.GraphGCN import GCN
import transformers
from layers.StandardNorm import Normalize
from layers.MTGNN  import gtnet  
from layers.AGCRN import AGCRN
from layers.ASTGCN  import ASTGCN
from layers.DMSTGCN import DMSTGCN
from layers.GMSDR import GMSDR
from layers.GMAN import GMAN
from layers.MSTGCN import MSTGCN
from layers.STSGCN import STSGCN
from layers.TGCN import TGCN
from layers.STGCN import STGCN


import torch.nn.functional as F
from copy import deepcopy
import numpy as np
import pandas as pd
import os
transformers.logging.set_verbosity_error()
import gc
from scipy.sparse.linalg import eigs

gc.collect()
torch.cuda.empty_cache()

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)   
        self.linear = nn.Linear(nf, target_window)   
        self.dropout = nn.Dropout(head_dropout)   

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x
    


class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.configs = configs
        # self.adj = torch.tensor(pd.read_csv(os.path.join(configs.root_path, configs.adjacency_path), 
        #                                     index_col=False,header=None).values).to(device='cuda:0', dtype=torch.float32)
        self.adj = pd.read_csv(os.path.join(configs.root_path, configs.adjacency_path), 
                                            index_col=False,header=None).values[1:,:]
        self.gcn = GCN(configs.seq_len, configs.seq_len, configs.seq_len)
        self.L_tilde = self.scaled_Laplacian(self.adj)
        self.cheb_polynomials = [torch.FloatTensor(i).to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')) for i in self.cheb_polynomial(self.L_tilde,K=3)]

        if configs.model == 'mtgnn':
            self.stmodel = gtnet(gcn_true=True, buildA_true=True,  gcn_depth=2,    
                            num_nodes=configs.node_num, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), 
                            predefined_A= self.adj, static_feat=None,  dropout=0.3,  subgraph_size=20,  
                            node_dim=40,  dilation_exponential=1,  conv_channels=32,   residual_channels=32,  skip_channels=64,   
                            end_channels=128,  seq_length= configs.seq_len,   in_dim=1,   out_dim=configs.seq_len, 
                            layers=3,   propalpha=0.05,  tanhalpha=3,  layer_norm_affline=True)
        elif configs.model == 'agcrn':
            self.stmodel =  AGCRN(num_node=configs.node_num,  input_dim= 1, hidden_dim=64,  output_dim = 1,
                            embed_dim = 64,  cheb_k = 2,   horizon = configs.seq_len,  num_layers = 2  )
            
        elif configs.model == 'astgcn':
            
            self.stmodel =  ASTGCN(DEVICE=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),cheb_polynomials = self.cheb_polynomials,  # 额外定义
                            num_for_prediction=configs.seq_len,   points_per_hour=configs.seq_len,  num_of_vertices = configs.node_num)
        
 

        elif configs.model == 'dmstgcn':
            self.stmodel = DMSTGCN(device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),num_nodes=configs.num_node, dropout=0.3, 
                            out_dim=configs.seq_len, residual_channels=32,  dilation_channels=32,  end_channels=512,  kernel_size=2, 
                            blocks=4,  layers=2,  days=288,  dims=configs.node_num,   order=2,  in_dim=2,  normalization="batch")
        
        elif configs.model == 'mstgcn':
            self.stmodel = MSTGCN(DEVICE=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),nb_block=3, in_channels= 1, 
                K=3,nb_chev_filter=64, nb_time_filter=64, cheb_polynomials=self.cheb_polynomials, num_for_prediction=configs.seq_len,
                points_per_hour=configs.seq_len)
        
        elif configs.model == 'stgcn':
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            L = self.scaled_laplacian(self.adj)
            Lk = self.cheb_poly(L, Ks=2)
            Lk = torch.Tensor(Lk.astype(np.float32)).to(device)
            self.stmodel = STGCN( ks=2, kt=3, bs=[[1, 16, 64], [64, 16, 64]], T=48, n=configs.node_num, Lk=Lk, p=0.)
            
        elif configs.model == 'gman':
            self.stmodel = GMAN( L = 1,  K =8,  d = 8,  num_his = configs.seq_len, bn_decay = 0.1,steps_per_day = 24,  use_bias = True, mask = True)
            self.stmodel = self.stmodel.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

        elif configs.model =='TGCN':
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.stmodel = TGCN(adj_mx=torch.tensor(self.adj, dtype=torch.float32).to(device), input_dim=1,  hidden_dim=64,  out_dim=configs.seq_len )
            self.stmodel = self.stmodel.to(device)

        else:
            print('configs.model is node defined!')
            raise ValueError

    def scaled_laplacian(self,A):
        n = A.shape[0]
        d = np.sum(A, axis=1)
        L = np.diag(d) - A
        for i in range(n):
            for j in range(n):
                if d[i] > 0 and d[j] > 0:
                    L[i, j] /= np.sqrt(d[i] * d[j])
        lam = np.linalg.eigvals(L).max().real
        return 2 * L / lam - np.eye(n)

    def cheb_poly(self,L, Ks):
        n = L.shape[0]
        LL = [np.eye(n), L[:]]
        for i in range(2, Ks):
            LL.append(np.matmul(2 * L, LL[-1]) - LL[-2])
        return np.asarray(LL)


    def cheb_polynomial(self, L_tilde, K):
        N = L_tilde.shape[0]
        cheb_polynomials = [np.identity(N), L_tilde.copy()]
        for i in range(2, K):
            cheb_polynomials.append(
                2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])
        return cheb_polynomials


    def scaled_Laplacian(self, W):
        assert W.shape[0] == W.shape[1]
        D = np.diag(np.sum(W, axis=1))
        L = D - W
        lambda_max = eigs(L, k=1, which='LR')[0].real
        return (2 * L) / lambda_max - np.identity(W.shape[0])



    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec,adj = None):
        if  self.configs.model =='mtgnn' :
            #len(x_enc.size()) == 4
            x_enc = torch.squeeze(x_enc)
            
            B, T, N = x_enc.size()
        #print('x_enc_pool:',x_enc_pool.shape)
        
            # Network adjacency matrix
            if adj is None:
                adj = torch.randn(N,N)
                adj = adj.to(x_enc.device)
            else:
                adj = self.adj.to(x_enc.device)        

            x_enc = torch.unsqueeze(x_enc, dim=-1)  #(B,F,N)——>(B,F,N,1)
            out = self.stmodel(x_enc.transpose(1,3)).to(torch.bfloat16) #adj)    #(B,F,N,1)——>(B,1,)  
            out = out.squeeze(-1)  # (B,F,N)    
        elif self.configs.model == 'stgcn':
            x_enc = torch.unsqueeze(x_enc, dim=-1)  #(B,F,N)——>(B,F,N,1)
            #out = self.stmodel(x_enc.transpose(1,3))
            x_enc = x_enc.to(torch.bfloat16)
            out = self.stmodel(x_enc).to(torch.bfloat16) #adj)    #(B,F,N,1)——>(B,1,)   
            out = out.squeeze(-1)  # (B,F,N)    
        elif self.configs.model == 'gman':
            x_enc = torch.unsqueeze(x_enc, dim=-1)  #(B,F,N)——>(B,F,N,1)
            #out = self.stmodel(x_enc.transpose(1,3))
            out = self.stmodel(x_enc).to(torch.bfloat16) #adj)    #(B,F,N,1)——>(B,1,)  
            out = out.squeeze(-1)  # (B,F,N)     
        elif self.configs.model == 'TGCN':
            x_enc = torch.unsqueeze(x_enc, dim=-1)  #(B,F,N)——>(B,F,N,1)
            #out = self.stmodel(x_enc.transpose(1,3))
            out = self.stmodel(x_enc).to(torch.bfloat16) #adj)    #(B,F,N,1)——>(B,1,)  
            out = out.squeeze(-1)  # (B,F,N)            
            
        else:
            x_enc = torch.unsqueeze(x_enc, dim=-1)
            B, T, N,D = x_enc.size()
            out = self.stmodel(x_enc)
            out = out.squeeze(-1)  # (B,F,N)   

        return out

