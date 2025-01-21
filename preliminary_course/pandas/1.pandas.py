import os
import torch

import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv(os.path.join('..', 'data', 'house_tiny.csv'))
    print(df)

    inputs, outputs = df.iloc[:, 0:2], df.iloc[:, 2]
    inputs = inputs.fillna(inputs.mean()) # 对于inputs中的数值进行平均数填充数
    print(inputs)

    # 对于inputs中的类别值或者离散值进行one-hot编码,
    # dummy_na=True表示对于缺失值也进行one-hot编码,
    # 将"NaN"视为有效的特征值并为其创建指示符变量
    inputs = pd.get_dummies(inputs, dummy_na=True)
    print(inputs)

    # 将pandas转换为tensor
    X, Y = torch.tensor(inputs.values), torch.tensor(outputs.values)
    print(X, Y)
