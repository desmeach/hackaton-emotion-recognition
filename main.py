import tensorflow as tf
import pandas as pd

dataset = pd.read_excel("dataset/dataset_train.xlsx", usecols=[4])
print(dataset)