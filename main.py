import tensorflow as tf
import pandas as pd
import keras as k
import numpy as np
from keras.layers import Dense, Activation

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
model = k.Sequential(([
        Dense(10, input_dim=inputs.shape[1]),
        Activation('relu'),
        Dense(5),
        Activation('softmax')
    ])
)
model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
fit_results = model.fit(x=inputs, y=outputs, epochs=10)

