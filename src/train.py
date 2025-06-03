from torch.autograd import Variable
import torch.nn as nn
import torch
from LSTMModel import lstm
from parser_my import args
from dataset import getData


def train():

    model = lstm(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size=1, dropout=args.dropout, batch_first=args.batch_first )
    model.to(args.device)
    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  

    _, _, train_loader, _ = getData(args.corpusFile,args.sequence_length,args.batch_size )

    for i in range(args.epochs):
        total_loss = 0
        for _, (data, label) in enumerate(train_loader):
            data1 = data.squeeze(1).cuda()
            pred = model(Variable(data1).cuda())
            pred = pred[1,:,:]
            label = label.unsqueeze(1).cuda()
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        

        if i % 100 == 0:
            print('%d epoch' % i)
            print(total_loss)

    torch.save({'state_dict': model.state_dict()}, args.save_file)

if __name__ == "__main__":
    train()