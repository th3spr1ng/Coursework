import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import os
import argparse
import tqdm
import random

import warnings
warnings.filterwarnings('ignore')


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        hidden_size = 100

        self.lstm = nn.RNN(input_size=4,
                           hidden_size=hidden_size,
                           batch_first=True,
                           num_layers=4,
                           dropout=0.1,
                           nonlinearity='relu',
                           bidirectional=False)

        self.linear = nn.Linear(in_features=hidden_size,
                                out_features=2)


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        _, h =  self.lstm(X)
        h = h.permute(1, 0, 2) # num_layers * num_directions, batch, hidden_size ->
                               # batch, num_layers * num_directions, hidden_size
        h = h.mean(dim=1)
        result = self.linear(h)
        return result



def preprocess_data(X: np.ndarray) -> np.ndarray:
    result = np.zeros(shape=(X.shape[0], 1000, 4))
    mapping = {
            'A': 0,
            'C': 1,
            'G': 2,
            'T': 3,
        }

    for line_idx, line in enumerate(X):
        for idx in range(len(line)):
            col_idx = mapping[line[idx]]
            result[line_idx, idx, col_idx] = 1

    return result



def train_model(X: np.ndarray, y: np.ndarray):
    model = Model()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), weight_decay=0.0001)

    epochs = 100
    batch_size = 32
    #  criterion = nn.CrossEntropyLoss(weight=torch.Tensor([0.05, 0.95]).to(device))
    criterion = nn.CrossEntropyLoss()

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.95,
                                                                random_state=42)
    X_train = torch.from_numpy(X_train).type(torch.float32).to(device)
    X_valid = torch.from_numpy(X_valid).type(torch.float32).to(device)
    y_train = torch.from_numpy(y_train).type(torch.long).to(device)
    y_valid = torch.from_numpy(y_valid).type(torch.long).to(device)

    train_length = len(X_train)
    valid_length = len(X_valid)

    for epoch_id in range(epochs):
        print('Epoch:', epoch_id)

        # Train loop
        model.train()

        for idx in range(0, train_length, batch_size):
            model.zero_grad()
            optim.zero_grad()

            batch = X_train[idx: idx + batch_size]
            target = y_train[idx: idx + batch_size]

            pred = model(batch)
            loss = criterion(pred, target)
            print('  Train loss:', loss.item())

            loss.backward()
            optim.step()

        # Validation loop
        #  model.eval()

        #  for idx in range(0, valid_length, batch_size):
        #      with torch.no_grad():
        #          batch = X_valid[idx: idx + batch_size]
        #          target = y_valid[idx: idx + batch_size]

        #          pred = model(batch)
        #          loss = criterion(pred, target)
        #          print('  Valid loss:', loss.item())

            os.makedirs('checkpoints/', exist_ok=True)
            torch.save(model.state_dict(), f'checkpoints/epoch_{epoch_id}.ckpt')



def run(filename: str, feature: str, target: str):
    df = pd.read_csv(filename)
    assert df[feature].isna().sum() == 0
    assert df[target].isna().sum() == 0

    X = np.array([x[:1000] for x in df[feature].values])
    y = df[target].values

    #  pos_idx = np.where(y == 1)[0]
    #  neg_idx = np.where(y == 0)[0]
    #  neg_idx = np.random.choice(neg_idx, pos_idx.shape[0], replace=False)

    #  X = np.concatenate([X[pos_idx], X[neg_idx]])
    #  y = np.concatenate([y[pos_idx], y[neg_idx]])

    X = preprocess_data(X)
    model = train_model(X, y)



if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, type=str, help='path to dataset')
    parser.add_argument('--dna-col', required=False, type=str, default='seq')
    parser.add_argument('--target-col', required=False, type=str, default='class')
    args = parser.parse_args()

    run(filename=args.data, feature=args.dna_col, target=args.target_col)

