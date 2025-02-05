import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveGate(nn.Module):
    def __init__(self, input_size):
        super(AdaptiveGate, self).__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, semantics_out, sthgnn_enc):
        # dec_out: (B, N, F)
        # sthgnn_enc: (B, N, F)
        combined_input = torch.cat((semantics_out, sthgnn_enc), dim=-1)  # (B, L, 2N)
        gate_values = self.gate_net(combined_input)  # (B, L, 2N)

        fused_output = gate_values * semantics_out + (1 - gate_values) * sthgnn_enc  # (B, N, F)
        #fused_output = dec_out_expanded + gate_values * torch.relu(mtgnn_enc)
        return fused_output
    

class LSTMGate(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMGate, self).__init__()
        self.lstm = nn.LSTM(input_size*2, hidden_size, batch_first=True)

    def forward(self, x, h):
        # x: (B, T, N)
        # h: (B, T, N)
        B, T, N = x.size()

        h = h.view(B, T, N).transpose(0, 1).contiguous()  # (T, B, N)
        h0 = h[-1].unsqueeze(0) # Use the last time step as the initial hidden state (1, B, N)
        c0 = torch.zeros_like(h0) # Initialize the cell state to zero (1, B, N)


        output, (h_n, c_n) = self.lstm(x, (h0, c0))

        output = output.view(B, T, N)  #Make sure the output shape is (B, T, N)

        return output
    


class AttentionGate(nn.Module):
    def __init__(self, input_size):
        super(AttentionGate, self).__init__()
        self.query = nn.Linear(input_size, input_size)
        self.key = nn.Linear(input_size, input_size)
        self.value = nn.Linear(input_size, input_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, dec_out, sthgnn_enc):
        # dec_out: (B, N, F)
        # sthgnn_enc: (B, N, F)
        query = self.query(dec_out)  # (B, N, F)
        key = self.key(sthgnn_enc)  # (B, N, F)
        value = self.value(sthgnn_enc)  # (B, N, F)

        scores = torch.bmm(query, key.transpose(1, 2)) / (query.size(-1) ** 0.5)  # (B, N, N)
        attention_weights = self.softmax(scores)  # (B, N, N)
        attended_sthgnn_enc = torch.bmm(attention_weights, value)  # (B, N, F)

        return attended_sthgnn_enc
