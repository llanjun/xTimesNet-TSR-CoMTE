import torch
import numpy as np
import ptwt
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd

def FFT_for_Period(x, k=2):
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)  
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    # return period, abs(xf).mean(-1)[:, top_list]
    return period, torch.softmax(abs(xf).mean(-1)[:, top_list], dim=1)

def CWT_for_Period(x, k=2, scales=None, wavelet='morl', plot=False):
    B, T, C = x.shape
    device = x.device
    if scales is None:
        max_scale = T // 2 + 1
        min_scale = 1
        scales = np.arange(min_scale, max_scale + 1)

        
    coeff, freq = ptwt.cwt(x.permute(0, 2, 1), scales, wavelet, sampling_period=1/T)
    coef_weight = abs(coeff).mean(1).mean(-1).mean(-1)
    _, topk_indices = torch.topk(coef_weight, k) 
    top_f = freq[topk_indices.cpu()]
    periods = (T//top_f).astype(int)
    cwt_amplitude = coeff[topk_indices, :, :, :].mean(-1).mean(-1).permute(1, 0)
    # return periods, cwt_amplitude
    return periods, torch.softmax(cwt_amplitude, dim=1)

def get_merged_periods(x, fft_k=3, cwt_k=3, scales=None, wavelet='morl', plot_cwt=False):

    fft_periods, fft_weights = FFT_for_Period(x, k=fft_k)

    cwt_periods, cwt_weights = CWT_for_Period(x, k=cwt_k, 
                                            scales=scales,
                                            wavelet=wavelet,
                                            plot=plot_cwt)
    

    merged_periods = list(set(fft_periods) | set(cwt_periods))
    
    fft_indices = [merged_periods.index(p) for p in fft_periods]
    cwt_indices = [merged_periods.index(p) for p in cwt_periods]
    
    combined_weights = torch.zeros(
        fft_weights.size(0), 
        len(merged_periods),
        device=fft_weights.device,
        dtype=fft_weights.dtype
    )
    combined_weights[:, fft_indices] += fft_weights
    combined_weights[:, cwt_indices] += cwt_weights
    
    merged_weights = torch.softmax(combined_weights, dim=1)
    
    

    return merged_periods, merged_weights


