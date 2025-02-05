import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
class AVWGCN(nn.Module):
    def __init__(self, in_dim, out_dim, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        """
            in_dim: The input dimension.
            out_dim: The output dimension.
            cheb_k: The order of Chebyshev polynomials, in the original text this parameter is set to 2.
            embed_dim: The embedding dimension of nodes.
        """
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, in_dim, out_dim))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, out_dim))
        init.xavier_uniform_(self.weights_pool)
        init.constant_(self.bias_pool, 0)
        
    def forward(self, x, node_embedding):
        """
            :param x: (B, N, C_in) - Where B is the batch size, N is the number of nodes, and C_in is the input channels.
            :param node_embedding: (N, D) - Node embeddings where N is the number of nodes and D is the embedding dimension. These are learnable parameters.
            :return: (B, N, C_out) - The output tensor with shape (Batch size, Number of nodes, Output channels).
        """
        num_node = node_embedding.shape[0]
        # adaptive adjacency matrix：adp = softmax(ReLU(E * E^T)), adp shape is (N, N)
        support = F.softmax(F.relu(torch.mm(node_embedding, node_embedding.transpose(0, 1))), dim=1)
        # The support computed here is a normalized Laplacian
        support_set = [torch.eye(num_node).to(support.device), support]  # A = [I, L]
        for k in range(2, self.cheb_k):
            # If cheb_k is set to 2, then this is not performed.
            # A(k) = 2 x L * A(k-1) - A(k-2)
            support_set.append(torch.matmul(2 * support, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)   # (K, N, N)
        # (N, D) * (D, K, C_in, C_out) -> (N, K, C_in, C_out)
        weights = torch.einsum('nd, dkio->nkio', node_embedding, self.weights_pool)
        # (N, D) * (D, C_out) -> (N, C_out)
        bias = torch.matmul(node_embedding, self.bias_pool)
        # GCN
        x_g = torch.einsum("knm, bmc->bknc", supports, x)  # (B, K, N, C_in)
        x_g = x_g.permute(0, 2, 1, 3)  # (B, N, K, C_in)
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias # (B, N, C_out)
        return x_gconv


class AGCRNCell(nn.Module):
    def __init__(self, num_node, in_dim, out_dim, cheb_k, embed_dim):
        super(AGCRNCell, self).__init__()
        self.num_node = num_node
        self.hidden_dim = out_dim
        # forget gate
        self.gate = AVWGCN(in_dim + out_dim, 2 * out_dim, cheb_k, embed_dim)
        # update gate
        self.update = AVWGCN(in_dim + out_dim, out_dim, cheb_k, embed_dim)

    def forward(self, x, state, node_embedding):
        # x: (B, N, C), state: (B, N, D)
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        # Generate two gates: forget gate and reset gate
        z_r = torch.sigmoid(self.gate(input_and_state, node_embedding))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embedding))
        h = r * state + (1 - r) * hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.num_node, self.hidden_dim)


class AVWDCRNN(nn.Module):
    def __init__(self, num_node, in_dim, out_dim, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >=1, "At least one DCRNN layer in the Encoder."
        self.num_node = num_node
        self.input_dim = in_dim
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(num_node, in_dim, out_dim, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(num_node, out_dim, out_dim, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embedding):
        """
        :param x: (B, T, N, in_dim)
        :param init_state: (num_layers, B, N, hidden_dim)
        :param node_embedding: (N, D)
        """
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embedding)
                inner_states.append(state)
            output_hidden.append(state)  # The hidden state output at the last time step of each layer
            current_inputs = torch.stack(inner_states, dim=1)  # (B, T, N, hidden_dim)

        # current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        # output_hidden: the last state for each layer: (num_layer, B, N, hidden_dim)
        output_hidden = torch.stack(output_hidden, dim=0)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []   # initial hidden state
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)  # (num_layer, B, N, hidden_dim)


class AGCRN(nn.Module):
    def __init__(self, num_node, input_dim, hidden_dim, output_dim, embed_dim, cheb_k, horizon, num_layers):
        super(AGCRN, self).__init__()
        self.num_node = num_node
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.horizon = horizon
        self.num_layers = num_layers

        # Learnable Node Embedding Vectors
        self.node_embedding = nn.Parameter(torch.randn(self.num_node, self.embed_dim), requires_grad=True)
        init.xavier_uniform_(self.node_embedding)
        # encoder
        self.encoder = AVWDCRNN(num_node, input_dim, hidden_dim, cheb_k, embed_dim, num_layers)
        # predictor
        self.end_conv = nn.Conv2d(1, self.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

    def forward(self, x):
        # x: (B, T, N, D)
        batch_size = x.shape[0]
        init_state = self.encoder.init_hidden(batch_size)
        output, _ = self.encoder(x, init_state, self.node_embedding)  # (B, T, N, D)
        output = output[:, -1:, :, :]   #  (B, 1, N, D)

        # predict layer
        output = self.end_conv(output)  # (B, T*1, N, 1)
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node) # (B, T, D, N)
        output = output.permute(0, 1, 3, 2)  # (B, T, N, D), D=1
        return output