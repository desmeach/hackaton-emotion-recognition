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
        Dense(40, input_dim=inputs.shape[1]),
        Activation('relu'),
        Dense(20),
        Activation('softmax')
    ])
)
model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
fit_results = model.fit(x=inputs, y=outputs, epochs=10)

test = [-24, 51, 75, 51, 14, -13, -25, -45, -77, -76, -18, 84, 146, 147, 113, 75, 48, 30, 13, -13, -32, 66, 152, 198, 181, 143, 101, 75, 56, 35, 6, -27, 52, 134, 195, 185, 135, 95, 63, 45, 11, -38, -61, -18, 83, 149, 149, 118, 62, 31, -2, -34, -66, -99, -90, -10, 73, 97, 62, 13, -20, -53, -84, -108, -139, -128, -65, 14, 36, 4, -47, -81, -99, -113, -142, -170, -133, -39, 11, 11, -29, -71, -98, -120, -148, -170, -149, -60, 10, 47, 25, -21, -63, -88, -107, -133, -142, -85, 8, 62, 55, 15, -19, -62, -93, -115, -140, -138, -69, 8, 36, 15, -22, -52, -79, -110, -132, -157, -138, -80, -30, -27, -63, -106, -136, -155, -183, -220, -233, -197, -147, -119, -135, -169, -202, -226, -245, -263, -282, -272, -224, -187, -185, -217, -246, -268, -275, -288, -308, -308, -268, -218, -204, -225, -256, -278, -286, -292, -306, -304, -268, -212, -186, -200, -229, -250, -257, -260, -266, -272, -227, -162, -144, -160, -185, -204, -214, -217, -229, -227, -170, -110, -89, -104, -125, -149, -157, -162, -175, -185, -145, -76, -38, -42, -68, -90, -103, -106, -120, -143, -144, -87, -27, 5, -2, -27, -50, -62, -68, -77, -90, -116, -114, -59, 9, 24, 9, -23, -45, -56, -67, -89, -108, -126, -127, -75, -19, 9, -8, -43, -63, -75, -86, -98, -116, -135, -138, -92, -43, -14]
test = [float(elem) for elem in test]
prediction = model.predict(test)
print(prediction)