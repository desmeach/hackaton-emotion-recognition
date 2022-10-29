import pandas as pd
import keras as k
import numpy as np
from keras.layers import Dense, Activation
from __future__ import absolute_import, division, print_function, unicode_literals
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
        tmp.append(np.array(arrayFromStr, dtype=float)[0:12])
    inputs.append(np.array(tmp))

for i, items in accum["Class_label_FPG"].items():
    outputs.append(np.array(items, dtype=float)[0:11])

inputs = np.array(inputs, dtype=object)
outputs = np.array(outputs, dtype=object)

model = tf.keras.Sequential()
model.add(layers.Embedding(input_dim=1000, output_dim=64))
model.add(layers.LSTM(128))
model.add(layers.Dense(10))
model.summary()
