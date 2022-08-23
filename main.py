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

print(

"""
headline_train, headline_test, category_train, category_test = train_test_split(headlines, categories, random_state=0, stratify=categories)

print(f'{headline_train[0]}, {category_train[0]}')

target_headline = headlines[0]

for word in target_headline: 
    if word != 0:
        for key, value in x_index.items():
            if value == word:
                print(f'{value}, {key}')
"""
