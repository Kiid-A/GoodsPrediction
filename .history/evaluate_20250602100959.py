from LSTMModel import lstm
from dataset import getData
from parser_my import args
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error


def eval():
    model = lstm(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size=1)
    model.to(args.device)
    checkpoint = torch.load(args.save_file)
    model.load_state_dict(checkpoint['state_dict'])
    preds = []
    labels = []
    min_price, max_price, train_loader, test_loader = getData(args.corpusFile, args.sequence_length, args.batch_size)
    for idx, (x, label) in enumerate(test_loader):
        x = x.squeeze(1).cuda()
        pred = model(x)
        list = pred.data.squeeze(1).tolist()
        preds.extend(list[-1])
        labels.extend(label.tolist())

    axis_label = []
    axis_pred = []
    axis_x = []
    axis_error = []

    for i in range(len(preds)):
        axis_pred.append(preds[i][0] * (max_price - min_price) + min_price)
        axis_label.append(labels[i] * (max_price - min_price) + min_price)

    error = mean_squared_error(axis_label, axis_pred)

    df = pd.read_csv(args.corpusFile).iloc[::-1]

    if 'date' in df.columns:
        df['date'] = df['date'].str.replace('年', '').str.replace('月', '').astype(int)

    for col in df.columns:
        if col != 'date':  
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for i in range(len(df) - len(preds), len(df)):        
        axis_x.append(str(int(df['date'][len(df)-i-1])%10000))
        axis_error.append(np.round(error / np.mean(axis_label), 5) * 10000)

  
    print(f"MSE: {error}")
    print(f"mean error percent: {np.round(error / np.mean(axis_label), 5) * 100}%")

    plt.figure(figsize=(16, 8))
    plt.title(f'Prediction on {parser.target} Price using LSTM')
    plt.plot(axis_x, axis_label, label='label')
    plt.plot(axis_x, axis_pred, label='prediction')
    # plt.plot(axis_x, axis_error)
    plt.legend(['lable', 'prediction', 'mean error percent(‱)'])
    plt.xlabel('Time(YYMMDD)')
    plt.ylabel('Price')
    plt.show()


eval()