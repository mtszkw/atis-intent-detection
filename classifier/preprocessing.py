import numpy as np
from sklearn import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def remove_stopwords_from_sentence(sentence, stopwords):
    return ' '.join([word for word in sentence.split() if word not in (stop_words)])    


def one_hot_encode_labels(y_train, y_test):
    y_train_ohe = np.array(y_train)
    y_test_ohe  = np.array(y_test)
    
    label_transformer = preprocessing.LabelBinarizer()
    label_transformer.fit(y_train_ohe)

    y_train_ohe = label_transformer.transform(y_train_ohe)
    y_test_ohe  = label_transformer.transform(y_test_ohe)
    
    return y_train_ohe, y_test_ohe


def tokenize_input_sequences_with_padding(X_train, X_test, padding):
    tokenizer = Tokenizer(num_words=None, lower=True, split=" ")
    tokenizer.fit_on_texts(X_train)
    
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test  = tokenizer.texts_to_sequences(X_test)
    
    X_train = pad_sequences(X_train, maxlen=padding).tolist()
    X_test = pad_sequences(X_test, maxlen=padding).tolist()
    
    return X_train, X_test, tokenizer.word_index
