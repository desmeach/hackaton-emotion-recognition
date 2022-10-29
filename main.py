import tensorflow as tf
import pandas as pd
import keras as k
import numpy as np

dataset = pd.read_excel("dataset/dataset_train.xlsx", skiprows=[0], usecols = [4, 5])
content = dataset.iterrows()
inputs = []
outputs = []
for i, row in content:
    temp = row.iloc[0].replace('[', '').replace(']', '').replace(' ', '').split(',')
    res = [float(ele) for ele in temp]
    inputs.append(res)
    outputs.append(row.iloc[1])
inputs = np.array(inputs)
outputs = np.array(outputs)
model = k.Sequential()
model.add(k.layers.Dense(units=1, activation="linear"))
model.compile(loss="mse", optimizer="sgd")
fit_results = model.fit(x=inputs, y=outputs, epochs=100)
