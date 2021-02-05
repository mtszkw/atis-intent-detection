import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from embeddings import read_glove_matrix
from metrics import acc
from model import build_model
from preprocessing import one_hot_encode_labels, remove_stopwords_from_sentence, tokenize_input_sequences_with_padding


def read_data(data_dir: str) -> (pd.DataFrame, pd.DataFrame):
    df_train = pd.read_csv(
        os.path.join(data_dir, 'atis_intents_train.csv'), header=None, names=['intent', 'sentence'])
    df_test = pd.read_csv(
        os.path.join(data_dir, 'atis_intents_test.csv'), header=None, names=['intent', 'sentence'])
    return df_train, df_test


df_train, df_test = read_data('../input/atis-airlinetravelinformationsystem/')

X_train, X_test = df_train['sentence'], df_test['sentence']
y_train, y_test = df_train['intent'], df_test['intent']

stop_words = stopwords.words("english")
X_train = [remove_stopwords_from_sentence(x, stop_words) for x in X_train]
X_test  = [remove_stopwords_from_sentence(x, stop_words) for x in X_test]

y_train, y_test = one_hot_encode_labels(y_train, y_test)

input_length = 25
X_train, X_test, word_index = tokenize_input_sequences_with_padding(X_train, X_test, padding=input_length)
X_train = np.asarray(X_train).astype(np.int32)
X_test  = np.asarray(X_test).astype(np.int32)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.15, random_state=2020)
print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape, y_test.shape)

embed_dim = 300
max_vocab_size = len(word_index) + 1
embed_matrix = read_glove_matrix('../input/glove42b300dtxt/glove.42B.300d.txt', max_vocab_size, embed_dim)

model = build_model(input_length, max_vocab_size, embed_dim, embed_matrix)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=50, verbose=2, batch_size=128)

preds = model.predict(X_test)
print(f'Test set accuracy: {acc(y_test, preds)}')
