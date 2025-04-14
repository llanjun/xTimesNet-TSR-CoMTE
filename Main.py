import numpy as np
import pandas as pd
import argparse
import os
import shutil
import io
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
from models import TimesNet_one_tensor_input
from models import Autoformer_one_tensor_input
from models import Dlinear_one_tensor_input
from models import PatchTST_one_tensor_input,PatchTST
from models import LSTM_one_tensor_input
from models import FNN_one_tensor_input
from models import xTimesNet_one_tensor_input
from mimic3benchmark import metrics_mimic
from mimic3models.in_hospital_mortality import utils
from mimic3benchmark.readers import InHospitalMortalityReader
from mimic3models.preprocessing import Discretizer, Normalizer
from utils.tools import EarlyStopping
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.utils.data
import matplotlib
from torch import optim
import time
import warnings
from sklearn.metrics import (roc_auc_score, precision_recall_curve, auc)
import seaborn as sns
warnings.filterwarnings('ignore')
import pickle
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.manifold import TSNE
import sklearn
import random
import datetime
from scipy.fft import fft

seed=666 #3407
torch.manual_seed(seed)
np.random.seed(seed)
np.random.seed(seed)
random.seed(seed)


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


new_result_mimic= []
def test(model, test_loader): #loaded best model
    preds = []
    trues = []
    model.eval()
    with torch.no_grad():
        for i, (batch_x, label) in enumerate(test_loader):
            batch_x = batch_x.float().to(device)
 
            label = label.to(device)

            outputs = model(batch_x)
            preds.append(outputs.detach())
            trues.append(label)
    preds = torch.cat(preds, 0)
    trues = torch.cat(trues, 0)
    if args.num_class == 1:
        ROC_AUC = roc_auc_score(trues.cpu().numpy(), preds.cpu().numpy(), multi_class="ovr")
        (precisions, recalls, _) = precision_recall_curve(trues.cpu().numpy(), preds.cpu().numpy())
        aucpr = auc(recalls, precisions)
        print('roc_auc: ', ROC_AUC)
        print('auc_pr: ', aucpr)
    if args.num_class == 2:
        predictions = preds.cpu().numpy()[:, 1]
        true = torch.argmax(trues, dim=1).cpu().numpy()
        result_mimic = metrics_mimic.print_metrics_binary(true, predictions, verbose=1)
        new_result_mimic.append(result_mimic)
        
    return result_mimic

    

def split_data(data_dir, listfile_path, train_size, test_size):
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    random.shuffle(all_files)  

    train_files = all_files[:train_size]
    test_files = all_files[train_size:train_size + test_size]
    vali_files = all_files[train_size + test_size:]


    train_folder = os.path.join(data_dir, 'train')
    test_folder = os.path.join(data_dir, 'test')
    vali_folder = os.path.join(data_dir, 'vali')

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(vali_folder, exist_ok=True)

    for file in train_files:
        shutil.copy(os.path.join(data_dir, file), os.path.join(train_folder, file))
    for file in test_files:
        shutil.copy(os.path.join(data_dir, file), os.path.join(test_folder, file))
    for file in vali_files:
        shutil.copy(os.path.join(data_dir, file), os.path.join(vali_folder, file))

    train_listfile = os.path.join(data_dir, 'train_listfile.csv')
    test_listfile = os.path.join(data_dir, 'test_listfile.csv')
    vali_listfile = os.path.join(data_dir, 'vali_listfile.csv')

    all_labels = pd.read_csv(listfile_path)

    train_labels = all_labels[all_labels['stay'].isin(train_files)]
    test_labels = all_labels[all_labels['stay'].isin(test_files)]
    vali_labels = all_labels[all_labels['stay'].isin(vali_files)]

    train_labels.to_csv(train_listfile, index=False)
    test_labels.to_csv(test_listfile, index=False)
    vali_labels.to_csv(vali_listfile, index=False)

    return train_folder, test_folder, vali_folder, train_listfile, test_listfile, vali_listfile


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
    args.pred_len = 0 
    args.num_class = 2  

    
    save_dir = r"Train_curves"  
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    

    merged_data_dir = r"data/merge/merge_data" 
    merged_listfile_path = r"data/merge/merge_listfile.csv"  
    

    train_size = 14681
    test_size = 3236
    vali_size = 3222
    num_runs = 5
    all_test_results = []
    
    if args.use_gpu and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
        device = torch.device("cuda")
    else:
        device = 'cpu'

    print(device)
    
    run_cv_results = ['Model', 'Run', 'acc', 'prec0', 'prec1', 'rec0', 'rec1', 'auroc', 'auprc', 'minpse']
    for run in range(num_runs):
        print(f"------------------- Run {run + 1} -------------------")
        data_file_path = 'data/Five_folds/data_'+str(run+1)    
        if reloaddata == True:
            train_folder, test_folder, vali_folder, train_listfile, test_listfile, vali_listfile = split_data(
                merged_data_dir, merged_listfile_path, train_size, test_size
            )
            train_reader = InHospitalMortalityReader(dataset_dir=train_folder, 
                                                     listfile=train_listfile, period_length=48.0)
            vali_reader = InHospitalMortalityReader(dataset_dir=vali_folder, 
                                                    listfile=vali_listfile, period_length=48.0)
            test_reader = InHospitalMortalityReader(dataset_dir=test_folder, 
                                                    listfile=test_listfile, period_length=48.0)

            discretizer = Discretizer(timestep=float(args.timestep),
                                      store_masks=True,
                                      impute_strategy='previous',
                                      start_time='zero')
            discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
    
            cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

    
            normalizer = Normalizer(fields=cont_channels)  # Normalize only cont_channels
            normalizer_state = args.normalizer_state
            if normalizer_state is None:
                normalizer_state = 'ihm_ts{}.input_str-{}.start_time-zero.normalizer'.format(args.timestep, args.imputation)
                normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
                normalizer.load_params(normalizer_state)
            
            train_data = utils.load_data(train_reader, discretizer, normalizer, small_part=False)
            vali_data = utils.load_data(vali_reader, discretizer, normalizer, small_part=False)
            test_data = utils.load_data(test_reader, discretizer, normalizer, small_part=False)
            

            train_x, train_y = train_data[0], train_data[1]
            vali_x, vali_y = vali_data[0], vali_data[1]
            test_x, test_y = test_data[0], test_data[1]
            

            train_y = np.array(train_y)
            vali_y = np.array(vali_y)
            test_y = np.array(test_y)
            

            columns_to_remove = list(range(4, 49)) + list(range(62, 66))
            train_x = np.delete(train_x, columns_to_remove, axis=2)
            vali_x = np.delete(vali_x, columns_to_remove, axis=2)
            test_x = np.delete(test_x, columns_to_remove, axis=2)
            

            train_x = torch.from_numpy(train_x).to(torch.float32)
            train_y = torch.from_numpy(train_y).to(torch.float32)
            vali_x = torch.from_numpy(vali_x).to(torch.float32)
            vali_y = torch.from_numpy(vali_y).to(torch.float32)
            test_x = torch.from_numpy(test_x).to(torch.float32)
            test_y = torch.from_numpy(test_y).to(torch.float32)
            

            enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
            enc.fit(np.concatenate((train_y, test_y, vali_y), axis=0).reshape(-1, 1))
            train_y = enc.transform(train_y.reshape(-1, 1)).toarray()
            test_y = enc.transform(test_y.reshape(-1, 1)).toarray()
            vali_y = enc.transform(vali_y.reshape(-1, 1)).toarray()
            
            train_y = torch.from_numpy(train_y)
            test_y = torch.from_numpy(test_y)
            vali_y = torch.from_numpy(vali_y)
                
            
            np.save(data_file_path + '/train_x.npy', train_x.cpu().numpy())
            np.save(data_file_path + '/test_x.npy', test_x.cpu().numpy())
            np.save(data_file_path + '/vali_x.npy', vali_x.cpu().numpy())
            np.save(data_file_path + '/train_y.npy', train_y.cpu().numpy())
            np.save(data_file_path + '/test_y.npy', test_y.cpu().numpy())
            np.save(data_file_path + '/vali_y.npy', vali_y.cpu().numpy())
            
        if reloaddata == False:
            train_x = torch.from_numpy(np.load(data_file_path + '/train_x.npy'))
            test_x = torch.from_numpy(np.load(data_file_path + '/test_x.npy'))
            vali_x = torch.from_numpy(np.load(data_file_path + '/vali_x.npy'))
            train_y = torch.from_numpy(np.load(data_file_path + '/train_y.npy'))
            test_y = torch.from_numpy(np.load(data_file_path + '/test_y.npy'))
            vali_y = torch.from_numpy(np.load(data_file_path + '/vali_y.npy'))
        # Create datasets and data loaders
        train_dataset = TensorDataset(train_x, train_y)
        vali_dataset = TensorDataset(vali_x, vali_y)
        test_dataset = TensorDataset(test_x, test_y)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        vali_loader = DataLoader(vali_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        
        if args.model == 'TimesNet':
            model = TimesNet_one_tensor_input.Model(args).float().to(device)
        elif args.model == 'DLinear':
            model = Dlinear_one_tensor_input.Model(args).float().to(device)
        elif args.model == 'FNN':
            model = FNN_one_tensor_input.Model(args).float().to(device)
        elif args.model == 'LSTM':
            model = LSTM_one_tensor_input.Model(args).float().to(device)
        elif args.model == 'Autoformer':
            model = Autoformer_one_tensor_input.Model(args).float().to(device)
        elif args.model == 'PatchTST':
            model = PatchTST_one_tensor_input.Model(args).float().to(device) # PatchTST PatchTST_one_tensor_input
        elif args.model == 'xTimesNet':
            model = xTimesNet_one_tensor_input.Model(args).float().to(device)

        print('Args in experiment:')
        print(args)

        criterion = nn.BCELoss()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)
        model_optim = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98))
        train_ROC_history = []
        train_PRC_history = []
        Vali_ROC_history = []
        Vali_PRC_history = []
        
        best_vali_auroc = 0
        best_model_path =  f'Train_curves/model_weight/{current_time}_{args.model}.pth'
        for epoch in range(args.train_epochs):    #model train
            print('--------------Epoch:{}--------------- '.format(epoch+1))
            iter_count = 0
            train_loss = []
            model.train()
            epoch_time = time.time()
            for i, (batch_x, label) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(device)     
                
                label = label.to(device)        
                outputs = model(batch_x)
                loss = criterion(outputs, label.float())  
                train_loss.append(loss.item())
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
                model_optim.step()
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            print('Train set')
            Train_result = test(model, train_loader)
            train_ROC_history.append(Train_result['auroc'])
            train_PRC_history.append(Train_result['auprc'])
            print('Vali set')
            Vali_result = test(model, vali_loader)
            Vali_ROC_history.append(Vali_result['auroc'])
            Vali_PRC_history.append(Vali_result['auprc'])
            if Vali_result['auroc'] > best_vali_auroc:
                best_vali_auroc = Vali_result['auroc']
                torch.save(model.state_dict(), best_model_path)
                print("!! Find best model !!")
