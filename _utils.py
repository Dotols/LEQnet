import os, sys, time
import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
# %matplotlib inline

DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
data_path = os.path.join(os.path.split(os.getcwd())[0], 'data/STEAD')
result_path = os.path.join(os.path.split(os.getcwd())[0], 'LEQnet_result')

class SeismicDataset(Dataset): 
    def __init__(self, xs, ys):
        self.x_data = xs
        self.y_data = ys

    def __len__(self): 
        return len(self.x_data)

    def __getitem__(self, idx): 
        x = self.x_data[idx]
        y = self.y_data[idx]
        return x, y

def load_dataset(params):
    print('DEVICE \t\t: ', DEVICE)
    print('pid \t\t: ', os.getpid())

    print('\n---------------------------------Data---------------------------------\n')
    
    if params['mode'] == 'preload':
        with open(os.path.join(data_path, 'labeled_dump_STEAD', params['file_name']), 'rb') as f:
            dataset = pickle.load(f)
            
        t_count = [data['p_label'] for data in dataset].count(1)
        f_count = [data['p_label'] for data in dataset].count(0)

        print('file name \t:', params['file_name'])
        print('data length \t:', len(dataset))
        print('T/F count \t:', t_count, '/', f_count)
        print('T/F ratio \t:', 100*t_count/(t_count+f_count), '%')
        
        def make_x_p_s(dataset):

            x = np.array([data['data'].T for data in dataset])
            if params['output'] == 'classification':
                y_p = np.array([np.array([data['p_label']]) for data in dataset])
                y_s = np.array([np.array([data['p_label']]) for data in dataset])
            else:
                y_p = np.array([data['p_time_label'] for data in dataset])[:,np.newaxis]
                y_s = np.array([data['s_time_label'] for data in dataset])[:,np.newaxis]
                
            x = torch.as_tensor(x.astype(np.float32))
            y_p = torch.as_tensor(y_p.astype(np.float32))
            y_s = torch.as_tensor(y_s.astype(np.float32))

            
            return x, y_p, y_s
        
        def train_test_valid_split(dataset):
            train, test = train_test_split(dataset, 
                                           test_size=0.3, 
                                           shuffle=True, 
                                           random_state=1004)
            valid = test[int(0.25*len(test)):]
            test = test[:int(0.25*len(test))]

            return train, test, valid
            
        x, y_p, y_s = make_x_p_s(dataset)

        train_x, valid_x, test_x  = train_test_valid_split(x)
        train_y_p, valid_y_p, test_y_p = train_test_valid_split(y_p)
        train_y_s, valid_y_s, test_y_s = train_test_valid_split(y_s)
        
    elif params['mode'] == 'generator':        
        with open(os.path.join(data_path, 'labeled_dump_STEAD', 'train_'+params['file_name']), 'rb') as f:
            train_dataset = pickle.load(f)
            
        with open(os.path.join(data_path, 'labeled_dump_STEAD', 'valid_'+params['file_name']), 'rb') as f:
            valid_dataset = pickle.load(f)
        
        t_count = [data['p_label'] for data in train_dataset+valid_dataset].count(1)
        f_count = [data['p_label'] for data in train_dataset+valid_dataset].count(0)
        
        print('file name \t:', params['file_name'])
        print('data length \t:', len(train_dataset) + len(valid_dataset))
        print('T/F count \t:', t_count, '/', f_count)
        print('T/F ratio \t:', 100*t_count/(t_count+f_count), '%')
        
        valid_test_split = int(len(valid_dataset)*params['train_valid_test_split'][1]/sum(params['train_valid_test_split'][1:]))
        
        def make_x_p_s(dataset):            
            x = np.array([data['data'].T for data in dataset])
            if params['output'] == 'classification':
                y_p = np.array([np.array([data['p_label']]) for data in dataset])
                y_s = np.array([np.array([data['s_label']]) for data in dataset])
            else:
                y_p = np.array([data['p_time_label'] for data in dataset])
                y_s = np.array([data['s_time_label'] for data in dataset])
                
            x = torch.as_tensor(x.astype(np.float32))
            y_p = torch.as_tensor(y_p.astype(np.float32))
            y_s = torch.as_tensor(y_s.astype(np.float32))
            
            return x, y_p, y_s
        
        train_x, train_y_p, train_y_s = make_x_p_s(train_dataset)
        valid_x, valid_y_p, valid_y_s = make_x_p_s(valid_dataset[:valid_test_split])
        test_x, test_y_p, test_y_s = make_x_p_s(valid_dataset[valid_test_split:])
        
    print('Train size \t:', len(train_x))
    print('Valid size \t:', len(valid_x))
    print('Test size \t:', len(test_x))

    print('\n------------------------------Data Loader------------------------------\n')

    train_loader = DataLoader(
        dataset=SeismicDataset(train_x, train_y_p), batch_size=500, shuffle=True
    )
    valid_loader = DataLoader(
        dataset=SeismicDataset(test_x, test_y_p), batch_size=500, shuffle=True
    )
    test_loader = DataLoader(
        dataset=SeismicDataset(valid_x, valid_y_p), batch_size=500, shuffle=True
    )

    for (X_train, y_train) in train_loader : 
        print('X_train shape: ', X_train.size() , ' \ttype : ', X_train.type())
        print('y_train shape: ', y_train.size() , ' \ttype : ', y_train.type())
        break
    
    dataset = {
        'source' : params['source'],
        'train_size' : len(train_x),
        'batch_size' : X_train.size()[0],
#         'window_size' : X_train.size()[2]//y_train.size()[2],
#         'window_count' : y_train.size()[2],
        'train_loader' : train_loader,
        'valid_loader' : valid_loader,
        'test_loader' : test_loader
    }
    
    return dataset

def metric(test_outputs, test_y_p):
    cnt = 0
    TP = FP = TN = FN = 0 + sys.float_info.epsilon
    for idx in range(len(test_y_p)):
        if int(torch.round(test_outputs[idx])) == 1 and int(torch.round(test_y_p[idx])) == 1:
            TP += 1
        if int(torch.round(test_outputs[idx])) == 1 and int(torch.round(test_y_p[idx])) == 0:
            FP += 1
        if int(torch.round(test_outputs[idx])) == 0 and int(torch.round(test_y_p[idx])) == 0:
            TN += 1
        if int(torch.round(test_outputs[idx])) == 0 and int(torch.round(test_y_p[idx])) == 1:
            FN += 1
            
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1_score = 2*(precision * recall)/(precision + recall)
    
    return accuracy, precision, recall, F1_score

def save_fig(source, model_name, epoch, plot_dict):
    output_dir = os.path.join(result_path, 'metric_log', source, model_name)
    
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
        
    plt.figure(figsize=(10, 5))
    plt.plot(plot_dict['losses'])
    plt.plot(plot_dict['val_losses'])
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(os.path.join(output_dir, model_name + '_' + str(epoch).zfill(3) + '_loss.png'))
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(plot_dict['acc'])
    plt.plot(plot_dict['prec'])
    plt.plot(plot_dict['rec'])
    plt.plot(plot_dict['f1'])
    plt.legend(['val_accuracy', 'val_precision', 'val_recall', 'val_F1_score'], loc='lower right')
    plt.savefig(os.path.join(output_dir, model_name + '_' + str(epoch).zfill(3) + '_metric.png'))
    plt.close()