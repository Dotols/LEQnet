import pickle
import os, sys, time

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset

import numpy as np
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split

from _utils import save_fig
DEVICE = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
data_path = os.path.join(os.path.split(os.getcwd())[0], 'data/STEAD')
result_path = os.path.join(os.path.split(os.getcwd())[0], 'LEQnet_result')

class CNNEncoder(nn.Module):
    def __init__(self, params):
        super(CNNEncoder, self).__init__()
        
        self.kernel_size = params['kernel_size']
        self.data_size = 1000
        self.embed_size = params['embed_size']
        
        self.conv1 = nn.Conv1d(3, 4, kernel_size=self.kernel_size, padding=self.kernel_size//2)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(4, 8, kernel_size=(self.kernel_size-2), padding=(self.kernel_size-2)//2)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(8, 16, kernel_size=(self.kernel_size-4), padding=(self.kernel_size-4)//2)
        self.maxpool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv1d(16, 32, kernel_size=(self.kernel_size-6), padding=(self.kernel_size-6)//2)
        self.maxpool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.embed = nn.Linear(in_features=32 * (self.data_size//16+1), out_features=self.embed_size)
        
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inp):
        inp = inp.to(DEVICE, dtype=torch.float)
        out = self.conv1(inp)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.maxpool1(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.maxpool2(out)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.maxpool3(out)

        out = self.conv4(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.maxpool4(out)
        
        out = out.view(-1, 32 * (self.data_size//16+1))
        out = self.embed(out)
        
        return out
    

class RNNDecoder(nn.Module):
    def __init__(self, params):
        super(RNNDecoder, self).__init__()
        
        self.embed_size = params['embed_size']
        self.hidden_size = params['hidden_size']
        
        self.lstm_cell = nn.LSTMCell(input_size = self.embed_size, hidden_size = self.hidden_size)
        self.fc_out = nn.Linear(in_features=self.hidden_size, out_features=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inp_emb):
        batch_size = inp_emb.size(0)
        window_count = inp_emb.size(1)//self.embed_size
        
        hidden_state = torch.zeros((batch_size, self.hidden_size)).to(DEVICE)
        cell_state = torch.zeros((batch_size, self.hidden_size)).to(DEVICE)
        
        outs = torch.empty((batch_size, window_count))
        
        for idx in range(window_count):
            hidden_state, cell_state = self.lstm_cell(inp_emb[:, idx*512:(idx+1)*512], (hidden_state, cell_state))
            
            out = self.fc_out(hidden_state)
            out = self.sigmoid(out)
            outs[:, idx] = out.reshape(-1)
            
        return outs
    
    
class C_LSTM(nn.Module):
    def __init__(self, h_params):
        super(C_LSTM, self).__init__()
        
        self.cell_count = h_params['window_count']
        self.inp_size = 6000//h_params['window_count']
        self.kernel_size = h_params['kernel_size']

        self.cnn_encoder = CNNEncoder(h_params).to(DEVICE)
        self.lstm_decoder = RNNDecoder(h_params).to(DEVICE)
        self.h_params = h_params

    def forward(self, inps):
        '''CNN-Encoder'''
        for cell in range(self.cell_count):
            inp = inps[:,:,cell*self.inp_size:(cell+1)*self.inp_size]

            embedding = self.cnn_encoder(inp)
            if cell == 0:
                embeddings = embedding
            else:
                embeddings = torch.cat((embeddings, embedding), 1)
        
        '''LSTM-Decoder'''
        outps = self.lstm_decoder(embeddings).to(DEVICE)
            
        return outps.reshape(-1, 1, 6)

def train(dataset, h_params):
    epochs = h_params['epochs']
    batch_size = dataset['batch_size']
    total_step = dataset['train_size']//batch_size
    c_lstm = C_LSTM(h_params).to(DEVICE)

    criterion = nn.BCELoss(reduction='mean')
    params = list(c_lstm.parameters())
    optimizer = torch.optim.Adam(params, lr=h_params['lr'])

    plot_dict = dict()
    losses = val_losses = acc = prec = rec = f1 = list()
    
    t0 = time.time()
    for epoch in range(1, epochs+1):

        for i_step in range(1, total_step+1):
            c_lstm.train()
            c_lstm.zero_grad()

            inps, labels = next(iter(dataset['train_loader']))
            
            labels = labels.to(DEVICE)
            outps = c_lstm(inps)
            
            loss = criterion(outps, labels)
            
            loss.backward()
            optimizer.step()

            # - - - Validate - - -
            with torch.no_grad():

                c_lstm.eval()
                
                val_inps, val_labels = next(iter(dataset['valid_loader']))
                
                val_labels = val_labels.to(DEVICE)
                val_outps = c_lstm(val_inps)
                
                val_loss = criterion(val_outps, val_labels)
                
                val_accuracy, val_precision, val_recall, val_F1_score, TP, FP, TN, FN = metric(val_outps, val_labels)
                
            val_losses.append(val_loss.item())
            losses.append(loss.item())
            acc.append(val_accuracy)
            prec.append(val_precision)
            rec.append(val_recall)
            f1.append(val_F1_score)
            
            stats_loss = 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Val Loss: %.4f, ' % \
                        (epoch, epochs, i_step, total_step, loss.item(), val_loss.item())
            stats_metric = 'accuracy : %.4f, precision : %.4f, recall : %.4f, F1_score : %.4f, ' % \
                            (val_accuracy, val_precision, val_recall, val_F1_score)
            stats_conf = 'TP : %d, FP : %d, TN : %d, FN : %d' % (TP, FP, TN, FN)
            stats = stats_loss + stats_metric + stats_conf
            print('\r', stats, end="")

        print('\r', stats)

        plot_dict['val_losses'] = val_losses
        plot_dict['losses'] = losses
        plot_dict['acc'] = acc
        plot_dict['prec'] = prec
        plot_dict['rec'] = rec
        plot_dict['f1'] = f1

        if epoch == 1 or epoch == 3 or epoch == 5 or epoch%10 == 0:
            save_fig(dataset['source'], h_params['model_name'], epoch, plot_dict)

    t1 = time.time()

    print('finished in {} seconds'.format(t1 - t0))
    
    return c_lstm

def test(model, dataset):
    criterion = nn.BCELoss(reduction='mean')
    with torch.no_grad():

        model.eval()

        test_inps, test_labels = next(iter(dataset['test_loader']))
        
        test_labels = test_labels.to(DEVICE)
        test_outps = model(test_inps)
        
        test_loss = criterion(test_outps, test_labels)
        test_accuracy, test_precision, test_recall, test_F1_score, TP, FP, TN, FN = metric(test_outps, test_labels)

    stats = 'Loss: %.4f, accuracy : %.4f, precision : %.4f, recall : %.4f, F1_score : %.4f' % \
            (test_loss.item(), test_accuracy, test_precision, test_recall, test_F1_score)
    stats_conf = 'TP : %d, FP : %d, TN : %d, FN : %d' % (TP, FP, TN, FN)

    print(stats)
    print(stats_conf)
    
#     return test_inps, test_outps.to('cpu'), test_labels.to('cpu')


def metric(test_outps, test_y_p):
    test_outps = torch.round(test_outps)
    test_y_p = torch.round(test_y_p)
    TP = FP = TN = FN = 0 + sys.float_info.epsilon
    
    for idx in range(test_y_p.size()[0]):
        for i in range(test_y_p.size()[2]):
            
            if 1 in test_y_p[idx][0]:
                if int(test_outps[idx][0][i]) != int(test_y_p[idx][0][i]):
                    FN += 1
                    break
                if i == (test_y_p.size()[2]-1):
                    TP += 1
            else:
                if int(test_outps[idx][0][i]) != int(test_y_p[idx][0][i]):
                    FP += 1
                    break
                if i == (test_y_p.size()[2]-1):
                    TN += 1
            
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1_score = 2*(precision * recall)/(precision + recall)
    
    return accuracy, precision, recall, F1_score, TP, FP, TN, FN