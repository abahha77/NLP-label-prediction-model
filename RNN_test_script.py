import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from nltk import RegexpTokenizer
from gensim.models import Word2Vec
import numpy as np

test_data = pd.read_csv('D:/Deep Project/Datasets/test _no_label.csv')

with open('D:/Deep Project/Models/Saved_Preprocessing_models/RNN/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

test_text = test_data['review_description']
test_text = test_text.apply(lambda x: tokenizer.tokenize(str(x)))

word2vec_model = Word2Vec.load('D:/Deep Project/Models/Saved_Preprocessing_models/RNN/word2vec_model.model')

test_embeddings = test_text.apply(lambda x: [word2vec_model.wv[word] for word in x if word in word2vec_model.wv])

with open('D:/Deep Project/Models/Saved_Preprocessing_models/RNN/max_sequence_length.pkl', 'rb') as f:
    max_sequence_length = pickle.load(f)

test_padded = sequence.pad_sequences(test_embeddings, maxlen=max_sequence_length, padding='post', dtype='float32')
test_filtered_array = np.array(test_padded)

model = load_model('D:/Deep Project/Models/Saved_Preprocessing_models/RNN/model.h5')

predictions = model.predict(test_filtered_array)
label_mapping = {0: -1, 1: 0, 2: 1}

predicted_labels = np.argmax(predictions, axis=1)
mapped_labels = np.array([label_mapping[label] for label in predicted_labels])

result = pd.DataFrame({"ID": test_data['ID'], "labels": mapped_labels})
result.to_csv('D:/Deep Project/Result_csv_files/Rnn_result.csv', index=False)
