import numpy as np


def read_glove_matrix(glove_txt_file: str, max_vocab_size, embedded_dim):
    embedded_index = dict()  
    with open(glove_txt_file, 'r', encoding='utf-8') as glove:
        for line in glove:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embedded_index[word] = vector

    glove.close()

    embedded_matrix = np.zeros((max_vocab_size, embedded_dim))
    for x, i in word_index.items():
        vector = embedded_index.get(x)
        if vector is not None:
            embedded_matrix[i] = vector

    return embedded_matrix