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

x_train, x_test, y_train, y_test = train_test_split(x_vals, y_vals, stratify=y_vals, random_state=0)

#print(x_train)

tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(x_train)
#sequences_x_train = 
print(x_train)
