from pandas import read_csv
import numpy as np
from torch.utils.data import DataLoader,Dataset
import torch
import pandas as pd
from torchvision import transforms
from parser_my import args


def getData(corpusFile, sequence_length, batchSize):
    price_data = read_csv(corpusFile)
    # price_data.drop('Unnamed: 0', axis=1, inplace=True)
    df = price_data

    if 'date' in df.columns:
        df['date'] = df['date'].str.replace('年', '').str.replace('月', '').astype(int)

    for col in df.columns:
        if col != 'date':  
            df[col] = pd.to_numeric(df[col], errors='coerce')

    min_price = df['price'].min()
    max_price = df['price'].max()


    df = df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

    dfY = df['price']
    dfX = df

    sequence = sequence_length
    X = []
    Y = []
    
    for i in range(df.shape[0] - sequence):
        X.append(np.array(dfX.iloc[i:(i + sequence)], dtype=np.float32))
        Y.append(np.array(dfY.iloc[(i + sequence)], dtype=np.float32))

    total_len = len(Y)
   
    pct = args.split_pct

    trainx, trainy = X[:int(pct * total_len)], Y[:int(pct * total_len)]
    testx, testy = X[int(pct * total_len):], Y[int(pct * total_len):]
    train_loader = DataLoader(dataset=Mydataset(trainx, trainy, transform=transforms.ToTensor()), batch_size=batchSize,
                              shuffle=True)
    test_loader = DataLoader(dataset=Mydataset(testx, testy), batch_size=batchSize, shuffle=False)
    return min_price, max_price, train_loader, test_loader


class Mydataset(Dataset):
    def __init__(self, xx, yy, transform=None):
        self.x = xx
        self.y = yy
        self.tranform = transform

    def __getitem__(self, index):
        x1 = self.x[index]
        y1 = self.y[index]
        if self.tranform != None:
            return self.tranform(x1), y1
        return x1, y1

    def __len__(self):
        return len(self.x)
