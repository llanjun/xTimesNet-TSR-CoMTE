import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.hidden_dim = 128
        self.lstm_layer = nn.LSTM(
                self.enc_in,
                self.hidden_dim,
                num_layers=2,
                bidirectional=False,
                batch_first=True,
            )
        self.final_layer = nn.Sequential(
                nn.Linear(self.hidden_dim, configs.num_class),
            )
    def forward(self, x_enc):
        x = self.lstm_layer(x_enc)[0][:, -1, :]  
        x = self.final_layer(x)    
        return torch.softmax(x, dim=1)
        
    