import torch.nn as nn
from torch.nn import init
import torch
import torch.nn.functional as F

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


