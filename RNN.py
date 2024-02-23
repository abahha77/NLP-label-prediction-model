import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from nltk import RegexpTokenizer
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec
from keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('wordnet')
nltk.download('stopwords')

data = pd.read_excel('D:/Deep Compition/Datasets/train.xlsx')

X = data['review_description']
Y = data['rating']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit(y_test)

tokenizer = RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')

x_train = x_train.apply(lambda x: tokenizer.tokenize(str(x)))
x_test = x_test.apply(lambda x: tokenizer.tokenize(str(x)))

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

word2vec_model = Word2Vec(sentences=x_train, vector_size=100, window=5, min_count=1, workers=4)
word2vec_model.save("word2vec_model.model")

x_train_embeddings = x_train.apply(lambda x: [word2vec_model.wv[word] for word in x if word in word2vec_model.wv])
x_test_embeddings = x_test.apply(lambda x: [word2vec_model.wv[word] for word in x if word in word2vec_model.wv])

max_sequence_length = max(len(max(x_train_embeddings, key=len)), len(max(x_test_embeddings, key=len)))

with open('max_sequence_length.pkl', 'wb') as f:
    pickle.dump(max_sequence_length, f)

x_train_padded = sequence.pad_sequences(x_train_embeddings, maxlen=max_sequence_length, padding='post', dtype='float32')
x_test_padded = sequence.pad_sequences(x_test_embeddings, maxlen=max_sequence_length, padding='post', dtype='float32')

model = Sequential()
model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(max_sequence_length, 100)))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(3, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
model.fit(x_train_padded, y_train, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

y_test_np = y_test.to_numpy()
accuracy = model.evaluate(x_test_padded, y_test_np, verbose=0)[1]
print(f'Test Accuracy: {accuracy}')
model.save("model.h5")
