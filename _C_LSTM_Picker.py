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

DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def save_fig(source, model_name, epoch, plot_dict):
    DIR = os.getcwd()
    output_dir = os.path.join(DIR, 'outputs', source, model_name)
    
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
        
    plt.figure(figsize=(10, 5))
    plt.plot(plot_dict['losses'])
    plt.plot(plot_dict['val_losses'])
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(os.path.join(output_dir, model_name + '_' + str(epoch).zfill(4) + '_loss.png'))
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(plot_dict['acc'])
    plt.plot(plot_dict['prec'])
    plt.plot(plot_dict['rec'])
    plt.plot(plot_dict['f1'])
    plt.legend(['val_accuracy', 'val_precision', 'val_recall', 'val_F1_score'], loc='lower right')
    plt.savefig(os.path.join(output_dir, model_name + '_' + str(epoch).zfill(4) + '_metric.png'))
    plt.close()

class CNNEncoder(nn.Module):
    def __init__(self, window_size, kernel_size, embed_size):
        super(CNNEncoder, self).__init__()
        
        self.kernel_size = kernel_size
        self.window_size = window_size
        self.embed_size = embed_size
        
        self.conv1 = nn.Conv1d(3, 4, kernel_size=kernel_size, padding=kernel_size//2)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(4, 8, kernel_size=(kernel_size-2), padding=(kernel_size-2)//2)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(8, 16, kernel_size=(kernel_size-4), padding=(kernel_size-4)//2)
        self.maxpool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv1d(16, 32, kernel_size=(kernel_size-6), padding=(kernel_size-6)//2)
        self.maxpool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.embed = nn.Linear(in_features=32 * (self.window_size//16+1), out_features=embed_size)
        
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

        out = out.view(-1, 32 * (self.window_size//16+1))
        out = self.embed(out)
        
        return out
    

class RNNDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(RNNDecoder, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        
        self.lstm_cell = nn.LSTMCell(input_size = embed_size, hidden_size = hidden_size)
        self.fc_out = nn.Linear(in_features=hidden_size, out_features=100)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inp_emb):
        batch_size = inp_emb.size(0)
        print(inp_emb.size(0))
        print(inp_emb.size(1))
        print(inp_emb.size(2))
        window_count = inp_emb.size(2)//self.embed_size
        
        hidden_state = torch.zeros((batch_size, self.hidden_size)).to(DEVICE)
        cell_state = torch.zeros((batch_size, self.hidden_size)).to(DEVICE)
        
        outs = torch.empty((batch_size, window_count))
        
        for idx in range(window_count):
            hidden_state, cell_state = self.lstm_cell(inp_emb[:, 0, idx*512:(idx+1)*512], (hidden_state, cell_state))
            
            out = self.fc_out(hidden_state)
            out = self.sigmoid(out)
            outs[:, idx] = out.reshape(-1)
        return outs.reshape(500, 100,-1)
    

class CNNDecoder(nn.Module):
    def __init__(self, window_size, kernel_size, embed_size):
        super(CNNDecoder, self).__init__()
        
        self.kernel_size = kernel_size
        self.window_size = window_size
        self.embed_size = embed_size
        
        self.conv1 = nn.Conv1d(16, 4, kernel_size=kernel_size, padding=kernel_size//2)
        self.Upsample1 = nn.Upsample(scale_factor=5)
        self.conv2 = nn.Conv1d(4, 1, kernel_size=(kernel_size-2), padding=(kernel_size-2)//2)
        self.Upsample2 = nn.Upsample(scale_factor=3)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.embed = nn.Linear(in_features=32 * (self.window_size//16+1), out_features=embed_size)
        
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inp):
        inp = inp.to(DEVICE, dtype=torch.float)
        out = self.conv1(inp)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.Upsample1(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.Upsample2(out)
        
        return out
    
cell_count = 6

def forward(dataset, inps, labels):
    for i in range(cell_count):
        inp = inps[:,:,i*(6000//cell_count):(i+1)*(6000//cell_count)]
        label = labels[:,:,i]
        outp = cnn_encoder(inp)
        if i == 0:
            outps = outp
        else:
            outps = torch.cat((outps, outp), 1)

    embedding = torch.unsqueeze(outps, 1).to(DEVICE)

    outps = lstm_decoder(embedding).to(DEVICE)
    labels = labels.to(DEVICE)

    for i in range(cell_count):
        if i == 0:
            loss = criterion(outps[:,:,i], labels[:,:,i])
        else:
            loss += criterion(outps[:,:,i], labels[:,:,i])

    return outps, loss
    

def metric(test_outps, test_y_p):
    
    TP = FP = TN = FN = 0 + sys.float_info.epsilon
    
    for idx in range(test_y_p.size()[0]):
        for i in range(test_y_p.size()[2]):
            
            if 1 in test_y_p[idx][0]:
                if int(torch.round(test_outps[idx][0][i])) != int(torch.round(test_y_p[idx][0][i])):
                    FN += 1
                    break
                if i == (test_y_p.size()[2]-1):
                    TP += 1
            else:
                if int(torch.round(test_outps[idx][0][i])) != int(torch.round(test_y_p[idx][0][i])):
                    FP += 1
                    break
                if i == (test_y_p.size()[2]-1):
                    TN += 1
            
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1_score = 2*(precision * recall)/(precision + recall)
    
    return accuracy, precision, recall, F1_score, TP, FP, TN, FN

def train(dataset, h_params):
    global cnn_encoder
    global lstm_decoder
    global cnn_decoder
    global criterion
    epochs = h_params['epochs']
    batch_size = dataset['batch_size']
    total_step = dataset['train_size']//batch_size
    cnn_encoder = CNNEncoder(dataset['window_size'], h_params['kernel_size'], h_params['embed_size']).to(DEVICE)
    lstm_decoder = RNNDecoder(h_params['embed_size'], h_params['hidden_size']).to(DEVICE)
    cnn_decoder = CNNDecoder(dataset['window_size'], h_params['kernel_size'], h_params['embed_size']).to(DEVICE)

    criterion = nn.BCELoss(reduction='mean')
    params = list(cnn_encoder.parameters())
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
            cnn_encoder.train()
            lstm_decoder.train()
            cnn_encoder.zero_grad()
            lstm_decoder.zero_grad()

            inps, labels = next(iter(dataset['train_loader']))
            _, loss = forward(dataset, inps, labels)
            
            loss.backward()
            optimizer.step()

            # - - - Validate - - -
            with torch.no_grad():

                cnn_encoder.eval()
                lstm_decoder.eval()
                
                val_inps, val_labels = next(iter(dataset['valid_loader']))
                val_outps, val_loss = forward(dataset, val_inps, val_labels)
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

def test(dataset):    
    with torch.no_grad():

        encoder.eval()
        decoder.eval()

        test_inps, test_labels = next(iter(dataset['test_loader']))
        test_outps, test_loss = forward(dataset, test_inps, test_labels)
        test_accuracy, test_precision, test_recall, test_F1_score, _, _, _, _ = metric(test_outps, test_labels)

    stats = 'Loss: %.4f, accuracy : %.4f, precision : %.4f, recall : %.4f, F1_score : %.4f' % \
            (test_loss.item(), test_accuracy, test_precision, test_recall, test_F1_score)

    print(stats)
    
    return test_inps, test_outps.to('cpu'), test_labels.to('cpu')



