from LSTMModel import lstm
from dataset import getData
from parser_my import args
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error


def eval(input_date=None, input_price=None):  # 新增输入参数，默认为 None
    model = lstm(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers, output_size=1)
    model.to(args.device)
    # 显式设置 weights_only=True 以避免未来警告
    checkpoint = torch.load(args.save_file, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    preds = []
    labels = []
    min_price, max_price, train_loader, test_loader = getData(args.corpusFile, args.sequence_length, args.batch_size)

    for idx, (x, label) in enumerate(test_loader):
        x = x.squeeze(1).cuda()
        pred = model(x)
        # 这里假设 pred 是一个二维张量，我们取最后一个元素
        pred_value = pred.data.squeeze().tolist()
        if isinstance(pred_value, list):
            preds.extend(pred_value[-1])
        else:
            preds.append(pred_value)
        labels.extend(label.tolist())

    axis_label = []
    axis_pred = []
    axis_x = []
    axis_error = []

    for i in range(len(preds)):
        # 修改这里，直接使用 preds[i]
        axis_pred.append(preds[i] * (max_price - min_price) + min_price)
        axis_label.append(labels[i] * (max_price - min_price) + min_price)

    error = mean_squared_error(axis_label, axis_pred)

    df = pd.read_csv(args.corpusFile).iloc[::-1]

    if 'date' in df.columns:
        df['date'] = df['date'].str.replace('年', '').str.replace('月', '').astype(int)

    for col in df.columns:
        if col != 'date':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for i in range(len(df) - len(preds), len(df)):
        axis_x.append(str(int(df['date'][len(df) - i - 1]) % 10000))
        axis_error.append(np.round(error / np.mean(axis_label), 5) * 10000)

    print(f"MSE: {error}")
    print(f"mean error percent: {np.round(error / np.mean(axis_label), 5) * 100}%")

    plt.figure(figsize=(16, 8))
    plt.title(f'Prediction on {args.target} Price using LSTM')
    plt.plot(axis_x, axis_label, label='label')
    plt.plot(axis_x, axis_pred, label='prediction')
    # plt.plot(axis_x, axis_error)
    plt.legend(['lable', 'prediction', 'mean error percent(‱)'])
    plt.xlabel('Time(YYMMDD)')
    plt.ylabel('Price')
    plt.show()
