import pandas as pd
from sklearn.model_selection import train_test_split
from nltk import RegexpTokenizer
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import pickle
from keras.preprocessing import sequence
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, Embedding, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.metrics import classification_report

import nltk

nltk.download('stopwords')

data = pd.read_excel('D:/Deep Compition/Datasets/train.xlsx')

X = data['review_description']
Y = data['rating']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=42)

label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

tokenizer = RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')
x_train_tokens = [tokenizer.tokenize(sentence) for sentence in x_train]
x_test_tokens = [tokenizer.tokenize(sentence) for sentence in x_test]

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

stopwords_list = list(stopwords.words('arabic'))
x_train_tokens = [[word for word in sentence if word.lower() not in stopwords_list] for sentence in x_train_tokens]
x_test_tokens = [[word for word in sentence if word.lower() not in stopwords_list] for sentence in x_test_tokens]

word2vec_model = Word2Vec(sentences=x_train_tokens, vector_size=100, window=5, min_count=1, workers=4)
word2vec_model.save("word2vec_model.model")

x_train_embeddings = [
    [word2vec_model.wv[word] if word in word2vec_model.wv else [0] * word2vec_model.vector_size for word in sentence]
    for sentence in x_train_tokens
]

x_test_embeddings = [
    [word2vec_model.wv[word] if word in word2vec_model.wv else [0] * word2vec_model.vector_size for word in sentence]
    for sentence in x_test_tokens
]

max_sequence_length = max(len(max(x_train_embeddings, key=len)), len(max(x_test_embeddings, key=len)))

with open('max_sequence_length.pkl', 'wb') as f:
    pickle.dump(max_sequence_length, f)

x_train_padded = sequence.pad_sequences(x_train_embeddings, maxlen=max_sequence_length, padding='post', dtype='float32')
x_test_padded = sequence.pad_sequences(x_test_embeddings, maxlen=max_sequence_length, padding='post', dtype='float32')


def transformer_model(max_len, embed_dim=32, num_heads=2, feedforward_dim=32, num_classes=3):
    inputs = Input(shape=(max_len, word2vec_model.vector_size))
    transformer_block = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
    transformer_block = GlobalAveragePooling1D()(transformer_block)
    outputs = Dense(feedforward_dim, activation="relu")(transformer_block)
    outputs = Dense(num_classes, activation="softmax")(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


model = transformer_model(max_sequence_length, embed_dim=100, num_heads=2, feedforward_dim=32, num_classes=3)
model.compile(optimizer=Adam(learning_rate=0.001), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

model.fit(x_train_padded, y_train, epochs=20, batch_size=32, validation_data=(x_test_padded, y_test))

loss, accuracy = model.evaluate(x_test_padded, y_test)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

y_pred = model.predict(x_test_padded)
y_pred_classes = y_pred.argmax(axis=1)
print("\nClassification Report:\n", classification_report(y_test, y_pred_classes, target_names=['-1', '0', '1']))
model.save("model.h5")