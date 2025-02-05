import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F
from sklearn.cluster import KMeans
from layers.AdaGNN import hgypergraph_constructor, graph_constructor
# from layers.HyperGNN import  HypergraphConvolution #HyperGraphAttention,
import math


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        # print(A.shape)
        # print(x.shape)
        x = torch.einsum('ncwl,vw->ncvl',(x,A))
        return x.contiguous()

class dy_nconv(nn.Module):
    def __init__(self):
        super(dy_nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,nvwl->ncwl',(x,A))   # BFNL, BNNL,
        return x.contiguous()
    

class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)


class prop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(prop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear(c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self,x,adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device).to(torch.bfloat16)
        d = adj.sum(1)
        h = x
        dv = d
        a = adj / dv.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
        ho = self.mlp(h)
        return ho


class mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha


    def forward(self,x,adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device).to(torch.bfloat16)
        # print(type(adj))
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho = self.mlp(ho)
        return ho



class dy_mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(dy_mixprop, self).__init__()
        self.nconv = dy_nconv()
        self.mlp1 = linear((gdep+1)*c_in,c_out)
        self.mlp2 = linear((gdep+1)*c_in,c_out)

        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
        self.lin1 = linear(c_in,c_in)
        self.lin2 = linear(c_in,c_in)


    def forward(self,x):
        x1 = torch.tanh(self.lin1(x))
        x2 = torch.tanh(self.lin2(x))
        adj = self.nconv(x1.transpose(2,1),x2)
        adj0 = torch.softmax(adj, dim=2)
        adj1 = torch.softmax(adj.transpose(2,1), dim=2)

        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,adj0)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho1 = self.mlp1(ho)


        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, adj1)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho2 = self.mlp2(ho)

        return ho1+ho2



class dilated_1D(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_1D, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2,3,6,7]
        self.tconv = nn.Conv2d(cin,cout,(1,7),dilation=(1,dilation_factor))

    def forward(self,input):
        x = self.tconv(input)
        return x



class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2,3,6,7]
        cout = int(cout/len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin,cout,(1,kern),dilation=(1,dilation_factor)))

    def forward(self,input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][...,-x[-1].size(3):]
        x = torch.cat(x,dim=1)
        return x





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
        x = torch.matmul(x, self.W)  # (B, L, N, F) ->(B, L, N, F)
        x = torch.einsum('nm, blnf->blmf', H, x)  # (N, M), (B, L, N, F') -> (B, L, M, F')
        # information transformation on hyperedges
        x = F.relu(torch.matmul(x,self.U))  # (B, L, M, F') -> (B, L, M, F')
        # Hyperedges to Node information aggregation
        x_out = torch.einsum('mn,blmf->blnf',  H.t(), x) #  (N, M), (B, L, M, F') -> (B, L, N, F')
        return x_out









class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:,idx,:], self.bias[:,idx,:], self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)



class sthypergnn(nn.Module):
    def __init__(self, gcn_true, hgcn_true, hgat_true, buildA_true,buildH_true, gcn_depth, 
                 num_nodes,num_hyperedges, device, predefined_A=None, static_feat=None, dropout=0.3, 
                 subgraph_size=20, node_dim=40, dilation_exponential=1,  conv_channels=32,
                   residual_channels=32, skip_channels=64, end_channels=128,  seq_length=12, 
                   in_dim=2, out_dim=12, layers=3, propalpha=0.05, tanhalpha=3,  layer_norm_affline=True, 
                   true_adj = 0, scale_hyperedges=10, alpha = 0.1, beta = 0.3, gamma = 0.5):
        super(sthypergnn, self).__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.hgcn_true = hgcn_true
        self.hgat_true = hgat_true
        self.buildH_true = buildH_true

        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.bfloat16))
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.bfloat16))
        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.bfloat16))
        self.param = self.gamma + self.alpha + self.beta


        self.num_nodes = num_nodes
        self.num_hyperedges = num_hyperedges
        self.scale_hyperedges = scale_hyperedges
        self.dropout = dropout
        self.predefined_A = predefined_A
         
        self.true_adj = torch.tensor(true_adj).to(torch.bfloat16)
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.gconv3 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        
        
        # construct first-order network module
        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat)


        # construct high-order hypergraph module
        self.hgc = hgypergraph_constructor(num_nodes,num_hyperedges, node_dim, device,scale_hyperedges = scale_hyperedges, alpha=tanhalpha,static_feat=static_feat,metric='knn')



        self.seq_length = seq_length
        kernel_size = 7
        if dilation_exponential>1:
            self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            self.receptive_field = layers*(kernel_size-1) + 1

        for i in range(1):
            if dilation_exponential>1:
                rf_size_i = int(1 + i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
            else:
                rf_size_i = i*layers*(kernel_size-1)+1
            new_dilation = 1
            for j in range(1,layers+1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)

                self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=residual_channels,
                                                 kernel_size=(1, 1)))
                
                if self.seq_length>self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels, 
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.seq_length-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.receptive_field-rf_size_j+1)))
                if self.gcn_true:
                    self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv3.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

                


                if self.seq_length>self.receptive_field:
                    # self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length),elementwise_affine=layer_norm_affline))
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline))
             
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                             out_channels=end_channels,
                                             kernel_size=(1,1),
                                             bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                             out_channels=out_dim,
                                             kernel_size=(1,1),
                                             bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)
        

        self.idx = torch.arange(self.num_nodes)
        self.device = device



    def forward(self, input, idx=None):
        seq_len = input.size(3)
        assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.seq_length<self.receptive_field:
            input = nn.functional.pad(input,(self.receptive_field-self.seq_length,0,0,0))

        # Spatial learning module
        # One-order interaction 
        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    adp = self.gc(self.idx) 
                else:
                    adp = self.gc(idx)
            else:
                adp = self.predefined_A.requires_grad_(True)
            adp = adp.to(torch.bfloat16)

        # High-oder interaction 
        if self.hgcn_true:
            if self.buildH_true:
                if idx is None: 
                    hadp = self.hgc(self.idx)
                else:
                    hadp = self.hgc(idx)
            else:
                hadp = self.predefined_H.requires_grad(True)
        
        
            hadp = hadp.to(torch.bfloat16)


        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        for i in range(self.layers):

            # TCN Module

            residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)

            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training).to(torch.bfloat16)
    
            s = x.to(torch.bfloat16)
            s = self.skip_convs[i](s)
            skip = s + skip
            
            # SHCN Module：Spatio-Temporal HGCN
            true_adj = self.true_adj.to(x.device)
            # GCN Module
            if self.gcn_true:
                x1 = self.gconv1[i](x, adp)+self.gconv2[i](x, adp.transpose(1,0))  # + self.gconv3[i](x, true_adj)   # BLNF*NN = BLNF
            else:
                x1 = self.residual_convs[i](x)

            # HGCN Module
            if self.hgcn_true:
                b,l,f,n = x.shape
                hypergraph_conv = HypergraphConvolution(in_channels=n, out_channels=n)
                x2 =  hypergraph_conv(x, hadp)
            else:
                x2 = self.residual_convs[i](x)

            # Fusion of GCN and HGCN

            if self.hgcn_true:
                x = self.gamma* x1 + (1-self.gamma)* x2.to(x1.device)
            else:
                x = x1

            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x,self.idx)
            else:
                x = self.norm[i](x,idx)
 
        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)        


        return x





# x = torch.randn(64, 96, 120)
# x = torch.unsqueeze(x,dim = -1)
# adj = torch.randn(120,120)

# layer = hgtnet(
#     gcn_true=True, 
#     hgcn_true = True,
#     buildA_true=True, 
#     buildH_true = True,
#     gcn_depth =2, 
#     num_nodes =120, 
#     num_hyperedges = 20,
#     device='cpu', 
#     predefined_A=adj, 
#     static_feat=None, 
#     dropout=0.3, 
#     subgraph_size=20, 
#     node_dim=40, 
#     dilation_exponential=1, 
#     conv_channels=32, 
#     residual_channels=32, 
#     skip_channels=64, 
#     end_channels=128, 
#     seq_length=96, 
#     in_dim=1, 
#     out_dim=12, 
#     layers=3, 
#     propalpha=0.05, 
#     tanhalpha=3, 
#     layer_norm_affline=True,
#     true_adj = adj )

# y = layer(x.transpose(1, 3))
# y = y+0
# print(y.size())  # torch.Size([64, 12, 170, 1])
