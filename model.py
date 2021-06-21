import numpy as np
import torch
import torch.nn as nn
from utils import data_reshape

class RNN_stacked_block(nn.Module):
    def __init__(self, input_size=75, hidden_size=512, num_classes=60):
        super().__init__()
        self.bi_lstm = nn.LSTM(
                            input_size, 
                            hidden_size, 
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x -> (N, L, input_size) 
        # temporal: (N, 100, 75)
        # spatial: (window_size: 25  L: 100)
        # chain -> (N*4, 25, 3*100/4)  traversal -> (N*4, 47, 3*100/4)
        x, (hn, cn) = self.bi_lstm(x) # -> (N, L, 512*2)
        x = torch.max(x, dim=1).values  # -> (N, 512*2)
        x = self.dropout(x)
        x = self.fc(x) # -> (N, 60)

        return x # -> (N, 60)

class RNN_hier_block(nn.Module):
    def __init__(self, part_hid_size=128, body_hid_size=512, num_classes=60):
        super().__init__()
        part_joints_num = [6, 6, 4, 4, 5]  # number of joints for each body part
        self.slice_idx = [0, 18, 36, 48, 60, 75] # split the joints into parts
        self.rnn_parts = nn.ModuleList()
        for i in range(5):
            self.rnn_parts.append(
                nn.LSTM(
                    part_joints_num[i] * 3, 
                    part_hid_size, 
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True)
            )
        self.rnn_body = nn.LSTM(
                            part_hid_size * 5 * 2,  # 5 parts, bidirectional
                            body_hid_size, 
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(body_hid_size * 2, num_classes)

    def forward(self, x):
        parts_out = []
        for i in range(5):
            x_temp, (hn, cn) = self.rnn_parts[i](x[:, :, self.slice_idx[i]:self.slice_idx[i + 1]])
            parts_out.append(x_temp)

        x = torch.cat(parts_out, dim=-1) # (N, L, 128*2) -> (N, L, 128*2*5)
        x, (hn, cn) = self.rnn_body(x)   # -> (N, L, 512*2)
        x = torch.max(x, dim=1).values   # -> (N, 512*2)
        x = self.dropout(x)
        x = self.fc(x)

        return x    # -> (N, 60)

# Temporal RNN 
class Temporal_RNN(nn.Module):
    def __init__(self, model_type="hierarchical", num_classes=60):
        super().__init__()
        assert model_type in ["stacked", "hierarchical"], "Input correct temporal model type!"

        if(model_type == "stacked"):
            self.rnn_temp = RNN_stacked_block(num_classes=num_classes)
        else:
            self.rnn_temp = RNN_hier_block(num_classes=num_classes)

    def forward(self, x):
        x = data_reshape(x, type="temporal")
        x = self.rnn_temp(x)

        return x    # -> (N, 60)

class Spatial_RNN(nn.Module):
    def __init__(self, seq_type="traversal", window_size=25, num_classes=60):
        super().__init__()
        assert seq_type in ["chain", "traversal"], "Input correct spatial seq type!"
        assert 100 % window_size == 0, "100 must be divided by window size"

        self.window_size = window_size
        self.seq_type = seq_type

        input_size = window_size * 3

        self.rnn_spac = RNN_stacked_block(input_size=input_size, num_classes=num_classes)

    def forward(self, x):
        x = data_reshape(x, type=self.seq_type, window_size=self.window_size)
        x = self.rnn_spac(x)
        
        return x    # -> (N, 60)

class Two_Stream_RNN(nn.Module):
    def __init__(self, model_type="hierarchical", seq_type="traversal", num_classes=60, modified=False):
        super().__init__()
        self.rnn_temp = Temporal_RNN(model_type=model_type, num_classes=num_classes)
        self.rnn_spac = Spatial_RNN(window_size=int(100/4), seq_type=seq_type, num_classes=num_classes)
        self.fc_gate = nn.Linear(num_classes * 2, 1)
        self.sigmod = nn.Sigmoid()
        self.modified = modified
        self.w = 0.9
    
    def forward(self, x):
        t = self.rnn_temp(x) # -> (N, 60)
        s = self.rnn_spac(x) # -> (N*100/25, 60) == (N*4, 60)
        s = torch.mean(s.view(t.shape[0], -1, t.shape[1]), dim=1)   # -> (N, 60)

        if not self.modified:
            score = t * self.w + s * (1 - self.w) # -> (N, 60)
        else:
            temp_gate = self.sigmod(self.fc_gate(torch.cat([t, s], dim=-1))) # -> (N, 60)
            spac_gate = self.sigmod(self.fc_gate(torch.cat([t, s], dim=-1))) # -> (N, 60)
            score = temp_gate * t + spac_gate * s # -> (N, 60)

        return score

if __name__ == "__main__":
    x = torch.randn((2, 100, 25, 3)).cuda()
    # model = Temporal_RNN(model_type="hierarchical").cuda()
    model = Two_Stream_RNN(model_type="hierarchical", seq_type="traversal", num_classes=60, modified=True).cuda()
    pred = model(x)
    print(pred.shape)
    


