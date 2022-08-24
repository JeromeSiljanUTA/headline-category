# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

df = pd.read_json('news_dataset.json', lines=True)

df.drop(columns=['link', 'authors', 'short_description', 'date'], inplace=True)

data = df.values.tolist()

x_vals = [entry[1] for entry in data]
y_vals = [entry[0] for entry in data]

vocab_max = 100000
tokenizer = Tokenizer(num_words=vocab_max, oov_token='<OOV>')

tokenizer.fit_on_texts(x_vals)
x_index = tokenizer.word_index

seq_x_index = tokenizer.texts_to_sequences(x_vals)
headlines = pad_sequences(seq_x_index, padding = 'post')

#print(x_index)
print(x_vals[0])
print(seq_x_index[0])
arr = [379, 361, 155, 1086, 2319, 6, 397, 241, 78, 85, 267, 189, 10, 304]
print(headlines[0])
for thing in headlines[0]:
    for key, value in x_index.items():
        if value == thing:
            print(f'{value}, {key}')


