import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(configs.enc_in * configs.seq_len,128)
        self.linear2 = nn.Linear(128,configs.num_class)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Sigmoid()
    def forward(self, x_enc):
        size = (x_enc.shape[0], x_enc.shape[-1]*x_enc.shape[-2])
        x = x_enc.reshape(size)
        x_st1 = self.linear1(x)
        x_st2 = self.linear2(x_st1)
        x_st2 = self.dropout(x_st2)
        return torch.softmax(x_st2, dim=1)