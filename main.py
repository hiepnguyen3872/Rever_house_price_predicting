from predict.predict_rental import predict_rental_value
from predict.predict_sell import predict_sell_value
import pandas as pd
import sys

def predict(data_path, output_file=None, rental_flag=False):
    data = pd.read_csv(data_path)
    if rental_flag:
        result = predict_rental_value(data)
    else:
        result = predict_sell_value(data)

    with open(output_file, 'w') as f:
        for value in result:
            f.write(str(value) + '\n')
    print("Success!")


if __name__ == "__main__":
    n = len(sys.argv)
    if (n < 3 or n >4):
        print("Invalid input")
    else:
        if n == 3:
            data_path = sys.argv[1]
            output_file = sys.argv[2]
            predict(data_path=data_path, output_file=output_file)
        else:
            data_path = sys.argv[1]
            output_file = sys.argv[2]
            rental_flag = True if int(sys.argv[3]) == 1 else False
            predict(data_path=data_path, output_file=output_file, rental_flag=rental_flag)
