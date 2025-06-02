from evaluate import eval
from train import train
from parser_my import args


goods = [
    'BaiCai', 'CaiJiao', 'ChengZi', 'DaDou', 'GengDao', 'GengMi', 'HuangGua', 'HuaShengRen',
    'MianHua', 'PingGuo', 'ShanDao', 'ShanMi', 'SiJiDou', 'XiangJiao', 'XiaoMai', 'XiHongShi',
    'YouCaiZi', 'YuMi'
]

for good in goods:
    args.target = good
    args.corpusFile = f'data/{good}.csv'
    args.save_file = f'model/{good}.pkl'
    print(f"Training and evaluating model for {args.target}...")
    
    train()eval()