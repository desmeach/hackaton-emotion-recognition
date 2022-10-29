import pandas as pd
import keras as k
import numpy as np
from keras.layers import Dense, Activation

df = pd.read_excel("dataset/dataset_train.xlsx")
tmp = df.groupby(['Obfuscated name', 'Presentation']).agg({'Data': list, 'Class_label_FPG': list})
accum = tmp[['Data', 'Class_label_FPG']]
inputs = []
outputs = []

for i, items in accum["Data"].items():
    tmp = []
    for item in items:
        arrayFromStr = item.replace('[', '').replace(']', '').replace(' ', '').split(',')
        arrayFromStr = [float(el) for el in arrayFromStr]
        tmp.append(np.array(arrayFromStr))
    inputs.append(np.array(tmp))

for i, items in accum["Class_label_FPG"].items():
    temp = [float(elem) for elem in items]
    outputs.append(np.array(temp))

inputs = np.array(inputs)
outputs = np.array(outputs)

model = k.Sequential(
    ([
        Dense(40),
        Activation('relu'),
        Dense(20),
        Activation('softmax')
    ])
)
model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
fit_results = model.fit(x=inputs, y=outputs, epochs=10)
