import argparse
import torch

parser = argparse.ArgumentParser()


target = 'CaiJiao'

# TODO 常改动参数
parser.add_argument('--target', default='CaiJiao', type=str)
parser.add_argument('--corpusFile', default=f'data/{target}.csv')
parser.add_argument('--gpu', default=0, type=int) 
parser.add_argument('--split_pct', default=0.85, type=float)
parser.add_argument('--epochs', default=500, type=int) 
parser.add_argument('--layers', default=2, type=int) 
parser.add_argument('--input_size', default=4, type=int) 
parser.add_argument('--hidden_size', default=64, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--sequence_length', default=4, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--useGPU', default=True, type=bool) 
parser.add_argument('--batch_first', default=True, type=bool) 
parser.add_argument('--dropout', default=0.01, type=float)
parser.add_argument('--save_file', default=f'model/{parser.target}.pkl') 


args = parser.parse_args()
args.useGPU = True
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
args.device = device