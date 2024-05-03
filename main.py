import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, metrics

from utils import *
from usad import *

file = "TCP_Station_1_IP_length_10000_10_interval_label"
print(file)
dataset = pd.read_csv(f'data/{file}.csv')

normal, attack = train_test_split(dataset, train_size=1000000, shuffle=False)
train_size = len(normal)

print(f"Train size: {train_size}")

# Normal
normal = normal.drop(["label"], axis=1)

min_max_scaler = preprocessing.StandardScaler()

x = normal.values
x_scaled = min_max_scaler.fit_transform(x)
normal = pd.DataFrame(x_scaled)

# Attack
labels = attack['label'].values
attack = attack.drop(["label"], axis=1)

x = attack.values 
x_scaled = min_max_scaler.transform(x)
attack = pd.DataFrame(x_scaled)

# Training
window_size = 20
print(f"Window Size: {window_size}")
windows_normal=normal.values[np.arange(window_size)[None, :] + np.arange(normal.shape[0]-window_size)[:, None]]
windows_attack=attack.values[np.arange(window_size)[None, :] + np.arange(attack.shape[0]-window_size)[:, None]]

BATCH_SIZE =  7919
N_EPOCHS = 100
hidden_size = 100

w_size=windows_normal.shape[1]*windows_normal.shape[2]
z_size=windows_normal.shape[1]*hidden_size

windows_normal_train = windows_normal[:int(np.floor(.8 *  windows_normal.shape[0]))]
windows_normal_val = windows_normal[int(np.floor(.8 *  windows_normal.shape[0])):int(np.floor(windows_normal.shape[0]))]

train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_normal_train).float().view(([windows_normal_train.shape[0],w_size]))
) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_normal_val).float().view(([windows_normal_val.shape[0],w_size]))
) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_attack).float().view(([windows_attack.shape[0],w_size]))
) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = UsadModel(w_size, z_size)
model = to_device(model,device)

history = training(N_EPOCHS,model,train_loader,val_loader)

'''
torch.save({
            'encoder': model.encoder.state_dict(),
            'decoder1': model.decoder1.state_dict(),
            'decoder2': model.decoder2.state_dict()
            }, f"models/{file}_{train_size}_{window_size}.pth")

checkpoint = torch.load(f"models/{file}_{train_size}_{window_size}.pth")

model.encoder.load_state_dict(checkpoint['encoder'])
model.decoder1.load_state_dict(checkpoint['decoder1'])
model.decoder2.load_state_dict(checkpoint['decoder2'])

'''

results=testing(model,test_loader)

windows_labels=[]
for i in range(len(labels)-window_size):
    windows_labels.append(list(np.int_(labels[i:i+window_size])))


y_test = [1.0 if (np.sum(window) > 0) else 0 for window in windows_labels ]

y_pred=np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(),results[-1].flatten().detach().cpu().numpy()])

threshold = 10
y_pred_ = np.zeros(y_pred.shape[0])
y_pred_[y_pred >= threshold] = 1

print(f"Alarms: {np.count_nonzero(y_pred_ == 1)}")

with open(f'scores/usad_{file}_{train_size}_{window_size}.dat', 'w') as f:
    for i in range(len(y_pred)):
        f.write(str(y_pred[i]) + ';' + str(y_test[i]) + ';' + str(y_pred_[i])  + '\n')