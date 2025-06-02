from evaluate import eval
from train import train



goods = [
    'BaiCai', 'CaiJiao', 'ChengZi', 'DaDou', 'GengDao', 'GengMi', 'HuangGua', 'HuaShengRen',
    'MianHua', 'PingGuo', 'ShanDao', 'ShanMi', 'SiJiDou', 'XiangJiao', 'XiaoMai', 'XiHongShi',
    'YouCaiZi', 'YuMi'
]

for good in goods:
    args.target = good
    args.corpusFile = f'data/{args.target}.csv'
    args.save_file = f'model/{args.target}.pkl'
    print(f"Training and evaluating model for {args.target}...")
eval()
train()