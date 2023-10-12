# Gerekli Kütüphanelerin Import Edilmesi
from __future__ import print_function
import io
import os
import sys
import string
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoin, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM, Bidirectional, Embedding

translator = str.maketrans('', '', string.punctuation)

# Verilerin Import Edilmesi

df = pd.read_csv("./lyrics.csv", sep="\t")
df.head()
pdf = pd.read_csv("./PoetryFoundationData.csv", quotechar='"')
pdf.head()
