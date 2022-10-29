import pandas as pd
# import keras as k
import numpy as np
from presentation import Presentation
from patient import Patient

# from keras.layers import Dense, Activation

df = pd.read_excel("dataset/dataset_train.xlsx")
tmp = df.groupby(['Obfuscated name', 'Presentation']).agg({'Data': list, 'Class_label_FPG': list})
accum = tmp[['Data', 'Class_label_FPG']]
input = []
output = []

for i, items in accum["Data"].items():
    for item in items:
        tmp = item.replace('[', '').replace(']', '').replace(' ', '').split(',')
        tmp = [float(elem) for elem in tmp]
        input.append(tmp)

for i, items in accum["Class_label_FPG"].items():
    for item in items:
        input.append(float(item))

# names = dataset['Obfuscated name'].items()
# datas = dataset['Data'].items()
# presentations = dataset['Presentation'].items()
# names_and_pres = dict()
# for i, name in names:
#     for j, presentation in presentations:
#         d_list = []
#         for k, data_list in datas:
#             d_list.append()
#         names_and_pres[name] = {
#             presentation: data_list
#         }
#
# print(names_and_pres)
# model = k.Sequential(
#     ([
#         Dense(40, input_dim=inputs.shape[1]),
#         Activation('relu'),
#         Dense(20),
#         Activation('softmax')
#     ])
# )
# model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# fit_results = model.fit(x=inputs, y=outputs, epochs=10)
# print(fit_results)
