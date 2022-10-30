from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import numpy as np
import tensorflow as tf
from sklearn import preprocessing

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
        patientsDict[labels[0]] = {1: [], 2: [], 3: [], 4: []}
    datas = []
    for data in row[0]:
        data = data.replace('[', '').replace(']', '').replace(' ', '').split(',')
        temp = preprocessing.normalize([np.array(data, dtype=float)])[0]
        datas.append(temp)
    datas = np.array(datas, dtype=float)
    row[1] = np.array(row[1], dtype=float)
    patientsDict[labels[0]][labels[1]] = (datas, row[1][:2])

# inputs = patientsDict[list(patientsDict.keys())[0]][1][0]
# outputs = patientsDict[list(patientsDict.keys())[0]][1][1]

inputs = []
outputs = []
for key in list(patientsDict.keys()):
    for presentation in patientsDict[key]:
        if len(patientsDict[key][presentation]) > 0:
            inputs.append(patientsDict[key][presentation][0][:2])
            outputs.append(patientsDict[key][presentation][1])

inputs = tf.stack(inputs)
outputs = tf.stack(outputs)
print(inputs)
print(outputs)
#
# model = Sequential()
# model.add(LSTM(100, input_shape=(2, 240), return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(200))
# model.add(Dropout(0.2))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# it_results = model.fit(x=inputs, y=outputs, epochs=10)
