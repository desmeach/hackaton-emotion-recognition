from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import keras as k
import numpy as np
import tensorflow as tf

df = pd.read_excel("dataset/dataset_train.xlsx")
groups = df.groupby(['Obfuscated name', 'Presentation'])
rows = groups.agg({'Data': list, 'Class_label_FPG': list}).iterrows()
# accum = tmp[['Data', 'Class_label_FPG']]

inputs = []
outputs = []

patientsDict = {}
presentationDict = {}

for labels, row in rows:
    if not labels[0] in patientsDict:
        patientsDict[labels[0]] = {1: [], 2: [], 3: []}
    datas = []
    for data in row[0]:
        data = data.replace('[', '').replace(']', '').replace(' ', '').split(',')
        datas.append(data)
    datas = np.array(datas, dtype=float)
    row[1] = np.array(row[1], dtype=float)
    patientsDict[labels[0]][labels[1]] = (datas, row[1])

# for i, items in accum["Data"].items():
#     tmp = []
#     for item in items:
#         arrayFromStr = item.replace('[', '').replace(']', '').replace(' ', '').split(',')
#         tmp.append(np.array(arrayFromStr[0:11], dtype=float))
#         tmp
#     inputs.append(tmp)

# for i, items in accum["Class_label_FPG"].items():
#     outputs.append(np.array(items[0:11], dtype=float))


inputs = patientsDict[list(patientsDict.keys())[0]][1][0]
outputs = patientsDict[list(patientsDict.keys())[0]][1][1]
# inputs = np.array(inputs)
# outputs = np.array(outputs)

model = tf.keras.Sequential()
model.add(k.layers.Embedding(input_dim=1000, output_dim=64))
model.add(k.layers.LSTM(128))
model.add(k.layers.Dense(10))

# model = tf.keras.Sequential()
# model.add(k.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(len(inputs), 0)))
# model.add(k.layers.MaxPooling2D((2, 2)))
# model.add(k.layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(k.layers.MaxPooling2D((2, 2)))
# model.add(k.layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(k.layers.Flatten())
# model.add(k.layers.Dense(64, activation='relu'))
# model.add(k.layers.Dense(10))

model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
fit_results = model.fit(x=inputs, y=outputs, epochs=10)
