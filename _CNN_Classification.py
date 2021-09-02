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
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
data_path = os.path.join(os.path.split(os.getcwd())[0], 'data/STEAD')
result_path = os.path.join(os.path.split(os.getcwd())[0], 'LEQnet_result')

class CNNEncoder(nn.Module):
    def __init__(self, params):
        super(CNNEncoder, self).__init__()
        
        self.kernel_size = params['kernel_size']
        self.data_size = 6000
        
        self.conv1 = nn.Conv1d(3, 4, kernel_size=self.kernel_size, padding=self.kernel_size//2)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(4, 8, kernel_size=(self.kernel_size-2), padding=(self.kernel_size-2)//2)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(8, 16, kernel_size=(self.kernel_size-4), padding=(self.kernel_size-4)//2)
        self.maxpool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv1d(16, 32, kernel_size=(self.kernel_size-6), padding=(self.kernel_size-6)//2)
        self.maxpool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.5)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=0.5)
        
        
        self.relu5 = nn.ReLU()
        self.relu6 = nn.ReLU()
        
        self.fc1 = nn.Linear(in_features=32 * (self.data_size//16), out_features=1500)
        self.fc2 = nn.Linear(in_features=1500, out_features= 100)
        self.fc3 = nn.Linear(in_features=100, out_features=1) 
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inp):
        inp = inp.to(DEVICE, dtype=torch.float)
        out = self.conv1(inp)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.maxpool1(out)
        
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.maxpool2(out)
        
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.dropout3(out)
        out = self.maxpool3(out)
        
        out = self.conv4(out)
        out = self.relu4(out)
        out = self.dropout4(out)
        out = self.maxpool4(out)
        
        out = out.view(-1, 32 * (self.data_size//16))
        
        out = self.fc1(out)
        out = self.relu5(out)
        out = self.fc2(out)
        out = self.relu6(out)
        out = self.fc3(out)
        
        out = self.sigmoid(out)
        
        return out
    
def metric(test_outputs, test_y_p):
    test_outputs = torch.round(test_outputs)
    test_y_p = torch.round(test_y_p)
    TP = FP = TN = FN = 0 + sys.float_info.epsilon
    for idx in range(len(test_y_p)):
        if test_outputs[idx] == 1 and test_y_p[idx] == 1:
            TP += 1
        if test_outputs[idx] == 1 and test_y_p[idx] == 0:
            FP += 1
        if test_outputs[idx] == 0 and test_y_p[idx] == 0:
            TN += 1
        if test_outputs[idx] == 0 and test_y_p[idx] == 1:
            FN += 1
            
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1_score = 2*(precision * recall)/(precision + recall)
    
    return accuracy, precision, recall, F1_score

def train(h_params, dataset):
    epochs = h_params['epochs']
    batch_size = dataset['batch_size']
    total_step = dataset['train_size']//batch_size
    encoder = CNNEncoder(h_params).to(DEVICE)

    criterion = nn.BCELoss(reduction='mean')
    params = list(encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=h_params['lr'])

    plot_dict = dict()
    losses = list()
    val_losses = list()
    acc = list()
    prec = list() 
    rec = list()
    f1 = list()

    t0 = time.time()

    for epoch in range(1, epochs+1):

        for i_step in range(1, total_step+1):
            encoder.train()
            encoder.zero_grad()

            inp, label = next(iter(dataset['train_loader']))
            inp = inp.to(DEVICE)
            label = label.to(DEVICE)
            outputs = encoder(inp).to(DEVICE)
            
            loss = criterion(outputs, label)

            loss.backward()
            optimizer.step()

            # - - - Validate - - -
            with torch.no_grad():

                encoder.eval()

                val_inp, val_label = next(iter(dataset['valid_loader']))
                val_inp = val_inp.to(DEVICE)
                val_label = val_label.to(DEVICE)
                val_outputs = encoder(val_inp).to(DEVICE)

                val_loss = criterion(val_outputs, val_label)
                val_accuracy, val_precision, val_recall, val_F1_score = metric(val_outputs, val_label)
                
            val_losses.append(val_loss.item())
            losses.append(loss.item())
            acc.append(val_accuracy)
            prec.append(val_precision)
            rec.append(val_recall)
            f1.append(val_F1_score)

            stats = 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Val Loss: %.4f accuracy : %.4f, precision : %.4f, recall : %.4f, F1_score : %.4f' %                     (epoch, epochs, i_step, total_step, loss.item(), val_loss.item(), val_accuracy, val_precision, val_recall, val_F1_score)

            print('\r', stats, end="")

        print('\r', stats)

        plot_dict['val_losses'] = val_losses
        plot_dict['losses'] = losses
        plot_dict['acc'] = acc
        plot_dict['prec'] = prec
        plot_dict['rec'] = rec
        plot_dict['f1'] = f1

        if epoch == 1 or epoch == 3 or epoch == 5 or epoch%10 == 0:
            save_fig(h_params['source'], h_params['model_name'], epoch, plot_dict)

    t1 = time.time()

    print('finished in {} seconds'.format(t1 - t0))

def test(dataset):
    with torch.no_grad():
        encoder.eval()

        test_inp, test_label = next(iter(dataset['test_loader']))
        test_inp = test_inp.to(DEVICE)
        test_label = test_label.to(DEVICE)
        test_outputs = encoder(test_inp).to(DEVICE)

        test_loss = criterion(test_outputs, test_label)
        test_accuracy, test_precision, test_recall, test_F1_score = metric(test_outputs, test_label)

    stats = 'Loss: %.4f, accuracy : %.4f, precision : %.4f, recall : %.4f, F1_score : %.4f' %                     (test_loss.item(), test_accuracy, test_precision, test_recall, test_F1_score)

    print(stats)