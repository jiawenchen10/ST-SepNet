
import math
import torch
import torch.nn as nn

# from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias = True, init='xavier' ):
        super(GraphConvolution,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features,out_features))
        
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias',None)
        
        if init =='uniform':
        # Uniform Initialization
            self.reset_parameter_uniform()  
        elif init == 'xavier':
            self.reset_parameter_xavier()
        elif init =='kaiming':
            self.reset_parameter_kaiming()
        else:
            raise NotImplementedError
    
    def reset_parameter_uniform(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        nn.init.uniform_(self.weight, -stdv, stdv)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -stdv, stdv)
        
    def reset_parameter_xavier(self):
        nn.init.xavier_uniform_(self.weight.data,gain=0.2)
        if self.bias is not None:
            nn.init.constant_(self.bias.data,0.0)
    
    def reset_parameter_kaiming(self):
        nn.init.kaiming_normal_(self.weight.data,a=0,mode = 'fan_in')
        if self.bias is not None:
            nn.init.constant_(self.bias.data,0.0)
    
    
    def forward(self, input,adj):
        '''
        :param: input_size  = (B,F,N)
        :param: adj_size = (feature_number, node_number)
        '''
        B, F, N = input.size()


        adj = adj + torch.eye(len(adj[0]), device=adj.device)

        deg_inv_sqrt = torch.pow(adj.sum(dim=-1).clamp(min=1.0), -0.5)
        deg_inv_sqrt = deg_inv_sqrt.unsqueeze(-1)  # D^{-1/2}(N,N)


        adj_normalized = deg_inv_sqrt * adj *  deg_inv_sqrt  # D^{-1/2}*A* D^{-1/2}

        demand_trandfered  = torch.einsum('bfn,fo->bno', input, self.weight) # (B,F,N)*(N,O) = (B,F,O)
        demand_stream = torch.einsum('nm,bmo->bno', adj_normalized, demand_trandfered)  #(N,F) (B,F,O) = (B,N,O)
        
        
        if self.bias is not None:
            return demand_stream + self.bias   # AX + b  
        else:
            return demand_stream
        
    
class GCN(nn.Module):
    def __init__(self, input_dimension, num_hidden, out_dimension, dropout_rate=0.5, init='xavier'):
        '''
        :param  input_dimensions is the dimension of input space
        :param  num_hidden is the dimension of the embedding space
        :param  out_dimension is the dimension of the output

        '''
        super(GCN, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=dropout_rate)
        
        self.GCN1 = GraphConvolution(input_dimension, num_hidden, init=init)
        self.GCN2 = GraphConvolution(num_hidden, out_dimension, init=init)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x ,adj):
        '''
        :param: x represents the data of time-stream network ()
        :param: adj is the structure of network (M,N)
        '''

        # Apply first GCN layer, followed by dropout and ReLU activation
        x_enc = F.relu(self.GCN1(x, adj))
        x_enc = self.dropout(x_enc)  # Apply dropout after ReLU
        
        # Second GCN layer
        x_enc = self.GCN2(x_enc.permute(0,2,1), adj)
        
        # Log softmax for output
        x = self.log_softmax(x_enc)
        x = x_enc + x   # Residual connnection
        
        return x 






'''
input = torch.rand(32,312,256)
adj = torch.rand(256,256)

input_dimension = 312
num_hidden = 256
output_dimension = 256
dropout = 0.1
init = 'uniform'

gcn = GCN(input_dimension=input_dimension,num_hidden=num_hidden,out_dimension =output_dimension,dropout=dropout,init=init)
output = gcn(input,adj)
print(output)


'''
    