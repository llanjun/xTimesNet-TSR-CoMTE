import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import numpy as np
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1, DilatedInceptionBlockV1
import tsfel
from models.learning_shapelets import ShapeletsDistBlocks
from tsflex.features import MultipleFeatureDescriptors, FeatureCollection
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
from utils.integrated_period import get_merged_periods


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_ff, configs.d_ff,
                                num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                                num_kernels=configs.num_kernels)
        )


    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = get_merged_periods(
            x,
            fft_k=self.k,    
            cwt_k=self.k,    
            wavelet='morl',
            plot_cwt=False
        )

        res = []
        
        
        # for i in range(self.k):
        for i in range(len(period_list)):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous() #(B, N, length // period, period)
            # 2D conv: from 1d Variation to 2d Variation
            # out = out.reshape(B, length // period *N, period)
            # out = self.conv(out).reshape(B, N, length // period, period)
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res



    
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TimesBlock(configs) for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, 
                                           configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.num_classes = configs.num_class
        self.act = F.gelu
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.3)
        self.TimesNet_projection = nn.Linear(configs.d_model * configs.seq_len, self.num_classes)
        


    def forward(self, x_enc):
        timesnet_input = x_enc
        
        enc_out = self.enc_embedding(timesnet_input, None)  # [B, T, C]
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        timesnet_output = self.act(enc_out)  # B, 48, embedding_size
        timesnet_output = self.dropout1(timesnet_output)
        timesnet_output = timesnet_output.reshape(timesnet_output.shape[0], -1)

        output = self.TimesNet_projection(timesnet_output)

        output = F.softmax(output, dim=1)
        return output