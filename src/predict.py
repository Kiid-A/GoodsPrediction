from evaluate import eval
from parser_my import args


def predict(input_date):
    predicted_price = eval(input_date)
    return predicted_price
    
if __name__ == "__main__":
    input_date = input("请输入要预测的日期（格式：YYYYMM）：")
    predicted_price = eval(input_date)
    print(f"预测的 {input_date} 的价格为: {predicted_price}")