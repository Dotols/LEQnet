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
        
        
        self.relu5 = nn.ReLU()
        self.relu6 = nn.ReLU()
        
        self.fc1 = nn.Linear(in_features=8 * 250, out_features=1)
#         self.fc1 = nn.Linear(in_features=32 * 375, out_features=1500)
#         self.fc2 = nn.Linear(in_features=1500, out_features= 100)
#         self.fc3 = nn.Linear(in_features=100, out_features=1) 
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
        
#         out = self.conv3(out)
#         out = self.relu(out)
#         out = self.dropout(out)
#         out = self.maxpool3(out)
        
#         out = self.conv4(out)
#         out = self.relu(out)
#         out = self.dropout(out)
#         out = self.maxpool4(out)
        
        out = out.view(-1, 8 * 250)
        
        out = self.fc1(out)
#         out = self.relu(out)
#         out = self.fc2(out)
#         out = self.relu(out)
#         out = self.fc3(out)
        
        out = self.sigmoid(out)
        
        return out
    
def train(h_params, dataset):
    epochs = h_params['epochs']
    batch_size = dataset['batch_size']
    total_step = dataset['train_size']//batch_size
    encoder = CNNEncoder(h_params).to(DEVICE)

    criterion = nn.BCELoss(reduction='mean')
    params = list(encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=h_params['lr'])

    plot_dict = dict()
    losses = val_losses = acc = prec = rec = f1 = list()
    
    t0 = time.time()
    for epoch in range(1, epochs+1):

        for i_step in range(1, total_step+1):
            encoder.train()
            encoder.zero_grad()

            inps, labels = next(iter(dataset['train_loader']))

            for i in range(6):
                inp = inps[:,:,i*1000:(i+1)*1000]
                label = labels[:,:,i]
                output = encoder(inp)
                if i == 0:
                    outputs = output
                else:
                    outputs = torch.cat((outputs, output), 1)
                    
            outputs = torch.unsqueeze(outputs, 1).to(DEVICE)
            labels = labels.to(DEVICE)
            
            for i in range(6):
                if i == 0:
                    loss = criterion(outputs[:,:,i], labels[:,:,i])
                else:
                    loss += criterion(outputs[:,:,i], labels[:,:,i])
#             loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # - - - Validate - - -
            with torch.no_grad():
                encoder.eval()
                val_inps, val_labels = next(iter(dataset['valid_loader']))

                for i in range(6):
                    val_inp = val_inps[:,:,i*1000:(i+1)*1000]
                    val_label = val_labels[:,:,i]
                    val_output = encoder(val_inp)
                    if i == 0:
                        val_outputs = val_output
                    else:
                        val_outputs = torch.cat((val_outputs, val_output), 1)
                val_outputs = torch.unsqueeze(val_outputs, 1).to(DEVICE)
                val_labels = val_labels.to(DEVICE)
                

                for i in range(6):
                    if i == 0:
                        val_loss = criterion(val_outputs[:,:,i], val_labels[:,:,i])
                    else:
                        val_loss += criterion(val_outputs[:,:,i], val_labels[:,:,i])
                        
                val_accuracy, val_precision, val_recall, val_F1_score, TP, FP, TN, FN = metric(val_outputs, val_labels)
                
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
            save_fig(h_params['source'], h_params['model_name'], epoch, plot_dict)

    t1 = time.time()

    print('finished in {} seconds'.format(t1 - t0))
    
    return encoder

def test(model, dataset):
    criterion = nn.BCELoss(reduction='mean')
    
    with torch.no_grad():
        model.eval()

        test_inps, test_labels = next(iter(dataset['test_loader']))
        
        for i in range(6):
            test_inp = test_inps[:,:,i*1000:(i+1)*1000]
            test_label = test_labels[:,:,i]
            test_output = model(test_inp)
            if i == 0:
                test_outputs = test_output
            else:
                test_outputs = torch.cat((test_outputs, test_output), 1)
        test_outputs = torch.unsqueeze(test_outputs, 1).to(DEVICE)
        test_labels = test_labels.to(DEVICE)

        for i in range(6):
            if i == 0:
                test_loss = criterion(test_outputs[:,:,i], test_labels[:,:,i])
            else:
                test_loss += criterion(test_outputs[:,:,i], test_labels[:,:,i])

                        
        test_accuracy, test_precision, test_recall, test_F1_score, TP, FP, TN, FN = metric(test_outputs, test_labels)

    stats = 'Loss: %.4f, accuracy : %.4f, precision : %.4f, recall : %.4f, F1_score : %.4f' % \
            (test_loss.item(), test_accuracy, test_precision, test_recall, test_F1_score)
    stats_conf = 'TP : %d, FP : %d, TN : %d, FN : %d' % (TP, FP, TN, FN)

    print(stats)
    print(stats_conf)
    
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