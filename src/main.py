import tkinter as tk
from tkinter import ttk
from evaluate import eval
from train import train
from parser_my import args
from predict import predict

# 商品列表
goods = [
    'BaiCai', 'CaiJiao', 'ChengZi', 'DaDou', 'GengDao', 'GengMi', 'HuangGua', 'HuaShengRen',
    'MianHua', 'PingGuo', 'ShanDao', 'ShanMi', 'SiJiDou', 'XiangJiao', 'XiaoMai', 'XiHongShi',
    'YouCaiZi', 'YuMi'
]


def run_prediction():
    selected_good = combo_box.get()
    input_date = entry_date.get()

    args.target = selected_good
    args.corpusFile = f'data/{selected_good}.csv'
    args.save_file = f'model/{selected_good}.pkl'
    print(f"Training and evaluating model for {args.target}...")
    # train()
    result = predict(input_date)
    formatted_price = "{:.2f}".format(result)
    result_text = f"预测 {input_date} 的 {selected_good} 价格为: {formatted_price} 元"
    result_label.config(text=result_text)


# 创建主窗口
root = tk.Tk()
root.title("Goods Prediction")

# 创建下拉框
combo_box = ttk.Combobox(root, values=goods)
combo_box.set(goods[0])
combo_box.pack(pady=10)

# 创建日期输入框和标签
label_date = tk.Label(root, text="请输入要预测的日期（格式：YYYYMM）：")
label_date.pack(pady=5)
entry_date = tk.Entry(root)
entry_date.pack(pady=5)

# 创建按钮
button = tk.Button(root, text="Run Prediction", command=run_prediction)
button.pack(pady=20)

result_label = tk.Label(root, text="")
result_label.pack(pady=10)

# 运行主循环
root.mainloop()