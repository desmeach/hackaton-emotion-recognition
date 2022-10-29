import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def get_groups():
    path = "dataset/dataset_train.xlsx"
    dataset = pd.read_excel(path, skiprows=[0], usecols=[0, 4, 5])
    content = dataset.iterrows()
    inputs = []
    outputs = []
    for i, row in content:
        # temp = purge(row.iloc[1])
        name = row.iloc[0]

        # res = [float(ele) for ele in temp]
    #     inputs.append(res)
    #     outputs.append(row.iloc[1])
    # inputs = np.array(inputs)
    # outputs = np.array(outputs)
    # return inputs, outputs


def purge(iloc):
    return iloc.replace('[', '').replace(']', '').replace(' ', '').split(',')


get_groups()
