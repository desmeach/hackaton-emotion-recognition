from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
import numpy as np
import tensorflow as tf
from sklearn import preprocessing

df = pd.read_excel("dataset/dataset_train.xlsx")
groups = df.groupby(['Obfuscated name', 'Presentation'])
rows = groups.agg({'Data': list, 'Class_label_FPG': list}).iterrows()

inputs = []
outputs = []

patientsDict = {}
presentationDict = {}

for labels, row in rows:
    if not labels[0] in patientsDict:
        patientsDict[labels[0]] = {1: [], 2: [], 3: [], 4: []}
    datas = []
    for data in row[0]:
        data = data.replace('[', '').replace(']', '').replace(' ', '').split(',')
        temp = preprocessing.normalize([np.array(data, dtype=float)])[0]
        datas.append(temp)
    accum = []
    datas = np.array(datas, dtype=float)
    row[1] = np.array(row[1], dtype=float)
    patientsDict[labels[0]][labels[1]] = (datas, row[1][:2])

inputs = []
outputs = []

for key in list(patientsDict.keys()):
    for presentation in patientsDict[key]:
        if len(patientsDict[key][presentation]) > 0:
            accum = np.mean(patientsDict[key][presentation][0], axis=0, dtype=float)
            inputs.append(accum[:2])
            outputs.append(patientsDict[key][presentation][1])

outputs = tf.stack(outputs)
inputs = tf.stack(inputs)
print(outputs[0])
print(inputs[0])
# model = Sequential(
#     ([
#         Dense(40, 'relu', input_shape=outputs[0].shape),
#         Dense(20, 'softmax'),
#     ])
# )
# model.compile(loss='huber_loss', optimizer='adam', metrics=["accuracy"])
# fit_results = model.fit(x=inputs, y=outputs, epochs=10)
# print(fit_results)
#
