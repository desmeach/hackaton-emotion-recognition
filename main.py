import pandas as pd
# import keras as k
import numpy as np
from presentation import Presentation
from patient import Patient

# from keras.layers import Dense, Activation

dataset = pd.read_excel("dataset/dataset_train.xlsx", skiprows=[0], usecols=[0, 2, 4, 5])
content = dataset.iterrows()
inputs = []
outputs = []
patients = set()
presentations = []
units = []

print(tmp[["Presentation", "Data"]])

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
