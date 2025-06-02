import tkinter as tk
from tkinter import ttk
from evaluate import eval
from train import train
from parser_my import args

# 商品列表
goods = [
    'BaiCai', 'CaiJiao', 'ChengZi', 'DaDou', 'GengDao', 'GengMi', 'HuangGua', 'HuaShengRen',
    'MianHua', 'PingGuo', 'ShanDao', 'ShanMi', 'SiJiDou', 'XiangJiao', 'XiaoMai', 'XiHongShi',
    'YouCaiZi', 'YuMi'
]


def run_prediction():
    selected_good = combo_box.get()

    args.target = selected_good
    args.corpusFile = f'data/{selected_good}.csv'
    args.save_file = f'model/{selected_good}.pkl'
    print(f"Training and evaluating model for {args.target}...")
    # train()
    # 传递输入的日期和价格
    eval(0, 0)


# 创建主窗口
root = tk.Tk()
root.title("Goods Prediction")

# 创建下拉框
combo_box = ttk.Combobox(root, values=goods)
combo_box.set(goods[0])
combo_box.pack(pady=10)

# 创建按钮
button = tk.Button(root, text="Run Prediction", command=run_prediction)
button.pack(pady=20)

# 运行主循环
root.mainloop()