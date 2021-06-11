import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import os
from model import Model, preprocess_data
import argparse

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


def check(checkpoint: str, data: str, feature: str, target: str):
    df = pd.read_csv(data)
    assert df[feature].isna().sum() == 0
    assert df[target].isna().sum() == 0

    X = np.array([x[:1000] for x in df[feature].values])
    X = preprocess_data(X)
    y = df[target].values

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.5,
                                                                random_state=42)

    X_train = torch.from_numpy(X_train).type(torch.float32).to(device)
    X_valid = torch.from_numpy(X_valid).type(torch.float32).to(device)

    model = Model()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        pred = torch.softmax(model(X_valid[:100]), dim=1)
        pred = pred.detach().cpu().numpy()

    hard_pred = np.argmax(pred, axis=1)
    print('Accuracy:', accuracy_score(y_valid[:100], hard_pred))
    print('F1-Score:', f1_score(y_valid[:100], hard_pred))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, type=str, help='path to checkpoint')
    parser.add_argument('--data', required=True, type=str, help='path to data')
    parser.add_argument('--dna-col', required=False, type=str, default='seq')
    parser.add_argument('--target-col', required=False, type=str, default='class')
    args = parser.parse_args()

    check(checkpoint=args.checkpoint, data=args.data, feature=args.dna_col, target=args.target_col)

