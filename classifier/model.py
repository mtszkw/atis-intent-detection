import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Input


def build_model(input_length, max_vocab_size, embed_dim, embed_matrix):
    model = Sequential()
    model.add(Input(shape=(input_length,), dtype='int32'))
    model.add(Embedding(max_vocab_size, embed_dim, input_length=input_length, weights=[embed_matrix], trainable=False))
    model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(8, activation='softmax'))
    print(model.summary())
    return model