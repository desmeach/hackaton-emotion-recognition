from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import keras as k
import numpy as np
from keras.layers import Dense, Activation
import collections
import tensorflow as tf
from tensorflow.keras import layers

df = pd.read_excel("dataset/dataset_train.xlsx")
tmp = df.groupby(['Obfuscated name', 'Presentation']).agg({'Data': list, 'Class_label_FPG': list})
accum = tmp[['Data', 'Class_label_FPG']]
inputs = []
outputs = []

for i, items in accum["Data"].items():
    tmp = []
    for item in items:
        arrayFromStr = item.replace('[', '').replace(']', '').replace(' ', '').split(',')
        if (len(arrayFromStr) < 11):
            end = [0] * (11 - len(arrayFromStr))
            arrayFromStr[len(arrayFromStr):] = end
        tmp.append(np.array(arrayFromStr[0:11], dtype=float))
        tmp
    inputs.append(tmp)

for i, items in accum["Class_label_FPG"].items():
    if (len(items) < 11):
            end = [0] * (11 - len(items))
            items[len(items):] = end
    outputs.append(np.array(items[0:11], dtype=float))

inputs = np.array(inputs)
outputs = np.array(outputs)

model = tf.keras.Sequential()
model.add(layers.Embedding(input_dim=1000, output_dim=64))
model.add(layers.LSTM(128))
model.add(layers.Dense(10))

model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
fit_results = model.fit(x=inputs.tolist(), y=outputs.tolist(), epochs=10)
