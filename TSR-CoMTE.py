import numpy as np
import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
from models import TimesNet_one_tensor_input
from models import Autoformer_one_tensor_input
from models import Dlinear_one_tensor_input
from models import PatchTST_one_tensor_input,PatchTST
from models import LSTM_one_tensor_input
from models import FNN_one_tensor_input
from models import xTimesNet_one_tensor_input
from mimic3benchmark  import metrics_mimic
from mimic3models.in_hospital_mortality import utils
from mimic3benchmark.readers import InHospitalMortalityReader
from mimic3models.preprocessing import Discretizer, Normalizer
from utils.tools import EarlyStopping
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
import torch.utils.data
import matplotlib
# matplotlib.use('QT5Agg')
from torch import optim
import time
import warnings
from sklearn.metrics import (roc_auc_score,precision_recall_curve,auc)
from TSInterpret.InterpretabilityModels.counterfactual.TSEvoCF import TSEvo
from TSInterpret.InterpretabilityModels.Saliency.TSR import TSR
import seaborn as sns
warnings.filterwarnings('ignore')
import random 
import torch
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def evaluate_counterfactuals(X, X_prime, V, model):
    assert X.shape == X_prime.shape, 
    B, T, C = X.shape
    
    L1_distance = np.sum(np.abs(X_prime - X)) / (B * T * C)
    L2_distance = np.sqrt(np.sum((X_prime - X)**2)) / (B * T * C)

    modified_values = np.sum(X_prime != X, axis=(1, 2)) 
    modification_ratio = np.mean(modified_values / (T * C))  

    violent_samples = []
    for i in range(B):
        v_features = (X_prime[i, :, V] != X[i, :, V])
        if np.any(v_features):
            violent_samples.append(i)
    violent_ratio = len(violent_samples) / B

    y_pred_proba = model(torch.from_numpy(X).float()).detach().numpy()
    y_pred_cf_proba = model(torch.from_numpy(X_prime).float()).detach().numpy()
    y_pred_class = np.argmax(y_pred_proba, axis=1)
    y_cf_class = np.argmax(y_pred_cf_proba, axis=1)

    success = 0
    for i in range(B):
        if y_cf_class[i] != y_pred_class[i]:
            success += 1
    success_rate = success / B

    cf_target_dis = 0
    for i in range(B):
        if y_cf_class[i]==0: 
            cf_target_dis += np.abs(y_pred_cf_proba[i, 1] - y_pred_proba[i, 1])
        if y_cf_class[i]==1: 
            cf_target_dis += np.abs(y_pred_cf_proba[i, 0] - y_pred_proba[i, 0])
    cf_target_dis_rate = cf_target_dis / B
    

    result = {
        "L1_distance": L1_distance,
        "L2_distance": L2_distance,
        "modification_ratio": modification_ratio,
        "violent_ratio": violent_ratio,
        "success_rate": success_rate,
        "cf_target_dis_rate": cf_target_dis_rate
    }
    
    return result

    
    
    
def get_args():
    parser = argparse.ArgumentParser(description='Classification')
    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='classification',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--model', type=str, required=False, default='xTimesNet')
    # data loader
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    # model define
    parser.add_argument('--top_k', type=int, default=3, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=3, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=27, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=16, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=2, help='output size')
    parser.add_argument('--d_model', type=int, default=32, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=2, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
    
    
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # optimization
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='Exp', help='exp description')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', 
                        help='use automatic mixed precision training', default=False) 
    
    
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')
    
    parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                        default=os.path.join(os.path.dirname(__file__), 'data/in-hospital-mortality/'))
    parser.add_argument('--target_repl_coef', type=float, default=0.0)
    parser.add_argument('--timestep', type=float, default=1.0,
                        help="fixed timestep used in the dataset")
    parser.add_argument('--normalizer_state', type=str, default=None,
                        help='Path to a state file of a normalizer. Leave none if you want to '
                             'use one of the provided ones.')
    parser.add_argument('--imputation', type=str, default='previous')
    parser.add_argument('--small_part', dest='small_part', action='store_true')
    parser.add_argument('--save_every', type=int, default=1,
                        help='save state every x epoch')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    import datetime
    reloaddata = False 
    print('reload data:', reloaddata)
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H_%M")
    args = get_args()
    args.model = 'xTimesNet' 
    args.seq_len = 48
    args.train_epochs = 30
    args.label_len = 48 
    args.pred_len = 0  # 'prediction sequence length', for classification, equals to 0
    args.num_class = 2  

    
    train_x = torch.from_numpy(np.load('train_x.npy'))
    train_y = torch.from_numpy(np.load('train_y.npy'))
    test_x = torch.from_numpy(np.load('test_x.npy'))
    test_y = torch.from_numpy(np.load('test_y.npy'))

    
    
    path = os.path.join(args.checkpoints)
    if not os.path.exists(path):
        os.makedirs(path)
    time_now = time.time()
    # if args.use_gpu and torch.cuda.is_available():
    #     os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    #     device = torch.device("cuda")
    # else:
    #     device = 'cpu'
    device = 'cpu'
    print(device)
    
    if args.model == 'xTimesNet':
        model = xTimesNet_one_tensor_input.Model(args).float().to(device)
        model.load_state_dict(torch.load('20250324_13_26.pth'))


    model.eval()
    with torch.no_grad():
        output = model(test_x.float()).detach().numpy()
    prob_0 = output[:, 0]
    mask = (prob_0 >= 0.4) & (prob_0 <= 0.6)
    selected_indices = np.where(mask)[0] 
    selected_samples = test_x[selected_indices,:,:]
    selected_samples_label = model(selected_samples.float()).detach().numpy()
    
    train_x_array=train_x.detach().numpy()
    train_y_array=train_y.detach().numpy()
    test_y_array=test_y.detach().numpy()
    test_x_array=test_x.detach().numpy()
    
    selected_samples = test_x_array[selected_indices, :, :]

    
    #TSR  
    selected_samples_label = np.argmax(model(torch.from_numpy(selected_samples).float()).detach().numpy(), axis=1)
    for i in range(0, selected_samples.shape[0]):
        print(i)
        int_mod=TSR(model.cpu(), train_x.shape[1], train_x.shape[-1], method='IG',mode='time') 
        TSR_exp=int_mod.explain(selected_samples[i:i+1], labels=selected_samples_label[i], TSR =True)
        # int_mod.plot(selected_samples[i:i+1], TSR_exp, heatmap = True, save = 'TSR.png')
        TSR_exp_sum = np.sum(TSR_exp, axis=0)
        if i ==0:
            TSR_Total = TSR_exp_sum
        if i !=0:
            TSR_Total = np.row_stack((TSR_Total, TSR_exp_sum))
    np.save('TSR.npy', TSR_Total)
    
    # COMTE
    from COMTE_Local.COMTECF_feature_vary import COMTECF_feature_constrain
    feature_to_vary=[2,10,11]
    print('\n')    
    print('feature_to_vary: ', feature_to_vary)
    COMTE_model= COMTECF_feature_constrain(model,(train_x.numpy(), train_y.numpy()),
                                            backend='PYT',mode='time', method= 'brute',
                                            number_distractors=5,
                                            feature_to_vary=feature_to_vary)
    
    for i in range(0, selected_samples.shape[0]):
        cf_array, cf_label_list = COMTE_model.explain(selected_samples[i:i+1])
        if i == 0 :
            CF = cf_array 
        if i != 0:
            CF = np.row_stack((CF, cf_array))
        org_label=model(torch.from_numpy(selected_samples[i:i+1]).float()) 
        cf_pred_label = model(torch.from_numpy(cf_array).float())
        print('\n')    
        print('org_label: ', org_label)
        print('cf_pred_label: ', cf_pred_label)

    
