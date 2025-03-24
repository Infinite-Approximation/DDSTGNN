import torch
import torch.nn as nn

class Forecast_MLP(nn.Module):
    def __init__(self, hidden_dim, fk_dim, **model_args):
        super().__init__()
        self.in_seq_len = model_args['in_seq_length']
        self.time_fc = nn.Conv2d(model_args['in_seq_length'], 1, kernel_size=(1,1))
        self.act = nn.LeakyReLU()
        self.feat_fc = nn.Linear(hidden_dim, fk_dim)
    def forward(self, x):
        B, T, N, F = x.shape
        # 首先进行填充
        padding_len = (self.in_seq_len - T) // 2
        pad = (0,0, 0,0, padding_len, padding_len)
        x = nn.functional.pad(x, pad, mode='constant', value=0)
        x = self.act(self.time_fc(x))
        x = self.feat_fc(x)
        return x