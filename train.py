import gensim, re
import numpy as np
import pandas as pd
import pickle
import os
from os import listdir, sep

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.callbacks import EarlyStopping

data_folder = "data1"

def txtTokenizer(texts, max_words=20000):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    word_index = tokenizer.word_index
    return tokenizer, word_index

def preProcess(sentences):
    text = [re.sub(r'([^\s\w]|_)+', '', sentence) for sentence in sentences if sentence != '']
    text = [sentence.lower().strip().split() for sentence in text]
    return text

def loadData(data_folder):
    texts = []
    labels = []

    for folder in listdir(data_folder):
        if folder != ".DS_Store":
            print("Loading category: ", folder)
            for file in listdir(data_folder + sep + folder):
                if file != ".DS_Store":
                    print("Loading file: ", file)
                    with open(data_folder + sep + folder + sep + file, 'r', encoding="utf-8") as f:
                        all_of_it = f.read()
                        sentences = all_of_it.split('.')
                        sentences = preProcess(sentences)

                        texts += sentences
                        labels += [folder] * len(sentences)  # Gán nhãn theo tên thư mục

    return texts, labels

if not os.path.exists(data_folder + sep + "data.pkl"):
    print("Data file not found, building it!")

    texts, labels = loadData(data_folder)

    # Kiểm tra phân phối nhãn trước khi one-hot encoding
    print("Label distribution before one-hot encoding:")
    print(pd.Series(labels).value_counts())

    tokenizer, word_index = txtTokenizer(texts)
    X = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(X)

    # Chuyển đổi nhãn thành one-hot encoded vector
    y = pd.get_dummies(labels)

    with open(data_folder + sep + "data.pkl", 'wb') as file:
        pickle.dump([X, y, texts], file)
else:
    print("Data file found, loading it!")
    with open(data_folder + sep + "data.pkl", 'rb') as file:
        X, y, texts = pickle.load(file)

# In thông tin sau khi tải dữ liệu từ tệp pickle
print("After loading raw data")
print("X shape:", X.shape)
print("Sample X:", X[10:30])
print("Sample y:", y[10:30])
print("Sample texts:", texts[10:30])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)

if not os.path.exists(data_folder + sep + "word_model.model"):
    word_model = gensim.models.Word2Vec(texts, vector_size=300, min_count=1, epochs=10)
    word_model.save(data_folder + sep + "word_model.model")
else:
    word_model = gensim.models.Word2Vec.load(data_folder + sep + "word_model.model")


embedding_matrix = np.zeros((len(word_model.wv.index_to_key) + 1, 300))
for i, word in enumerate(word_model.wv.index_to_key):
    embedding_matrix[i + 1] = word_model.wv[word]

if not os.path.exists(data_folder + sep + "3epoch.keras"):
    model = Sequential()
    model.add(Embedding(len(word_model.wv.index_to_key) + 1, 300, input_length=X.shape[1], weights=[embedding_matrix],
                        trainable=False))
    model.add(LSTM(300, return_sequences=False, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(y.shape[1], activation="softmax"))
    model.summary()
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['acc'])

    batch_size = 64
    epochs = 3  # Có thể điều chỉnh số epoch nếu cần

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

    # Sử dụng phần mở rộng .keras hoặc .h5 để lưu mô hình
    model.save(data_folder + sep + "3epoch.keras")
else:
    model = load_model(data_folder + sep + "3epoch.keras")

model.evaluate(X_test, y_test)
