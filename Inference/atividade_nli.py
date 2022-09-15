# -*- coding: utf-8 -*-
"""Atividade_NLI.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rZ8VcCMX4DZoR-86lIW5_VPRWu7Hkjqx

# Imports and Downloads

1. Texts to sequences
2. pad sequences (consultar cnn)
"""

import pandas as pd
import numpy as np
import tensorflow_datasets as tfds

import requests
import re
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
# from keras.layers import Dense,Conv1D,Embedding,GlobalMaxPooling1D,MaxPooling1D,Activation,Flatten
# from sklearn.feature_extraction.text import TfidfVectorizer
# from tensorflow.keras.optimizers import SGD
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from keras.utils.np_utils import to_categorical
# from sklearn.metrics import classification_report
# import matplotlib.pyplot as plt

def conc(x,y):
  return x + y

def normalizarString(text):
    import unicodedata

    
    text = text.lower()
    text = re.sub(r'\'?b\'?','',text)
    text = re.sub(r'[\'\"]$','',text)
    return str(text.lower())


def usingStopwords():
    master = "https://raw.githubusercontent.com/guico3lho/NLP_UnB_2022_1/main/stopwords_pt.txt"
    req = requests.get(master)
    text = req.text

    stop_words = []
    pattern = r'\n?'
    rt = re.compile(pattern)

    for i in text.split('\n'):
        stop_words.append(normalizarString(rt.sub('', i)))
    print(f'Fim do Preenchimento das stopwords')

    return stop_words

stopwords = usingStopwords()

stopwords

ds_train =tfds.load('snli',split='train',shuffle_files=True)
ds_validation =tfds.load('snli',split='validation',shuffle_files=False)
ds_test =tfds.load('snli',split='test',shuffle_files=False)

len(ds_train)

# df_train = tfds.as_dataframe(ds_train)
# del df_train['similarity']
# df_train['concatenated'] = df_train.apply(lambda x: conc(str(x['text']),str(x['hypothesis'])),axis=1)
# df_model_train = df_train[['concatenated','entailment']]
# df_m
# df_model_train['concatenated'] = df_model_train['concatenated'].apply(lambda x: normalizarString(x))
# df_model_train

df_train = tfds.as_dataframe(ds_train.take(10000))

df_train.head(20)

"""# Training pre-processing"""

#@ concatening premise + hypotesis
df_train['premise_hyp'] = df_train.apply(lambda x: conc(str(x['premise']),str(x['hypothesis'])),axis=1)
df_model_train = df_train[['premise_hyp','label']]
df_model_train

#@ normalize texts
df_model_train['premise_hyp'] = df_model_train['premise_hyp'].apply(lambda x: normalizarString(x))
df_model_train

# df_model_train['label'] = pd.Categorical(df_model_train['label'])
# y_train_int = df_model_train['label'].cat.codes
# y_train = to_categorical(y_train_int)
# y_train.shape
# df_model_train.head(20)

train_sentences = []
train_labels = []
for i,row in df_model_train.iterrows():
  train_sentences.append(row[0])
  train_labels.append(row[1])

train_sentences

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocabulary = {}
for i in range(0,len(df_model_train)):
  for word in train_sentences[i].split():
    if word not in vocabulary:
      vocabulary[word] = 1
    else:
      vocabulary[word] += 1

tokenizer = Tokenizer(num_words = vocab_size)
tokenizer.fit_on_texts(df_model_train['premise_hyp'])

word_index = tokenizer.word_index

# print(f"{word_index['his']}")

vocab_size = len(vocabulary)
embedding_dim=16
max_length = 120
trunc_type= 'post'
padding_type = 'post'
oov_tok=""

sequences = tokenizer.texts_to_sequences(train_sentences)
padded = pad_sequences(sequences,maxlen=max_length,truncating = trunc_type,padding=padding_type)

padded[1]

"""# Validation Pre processing"""

df_valid = tfds.as_dataframe(ds_validation)
del df_valid['similarity']
df_valid['concatenated'] = df_valid.apply(lambda x: conc(str(x['text']),str(x['hypothesis'])),axis=1)
df_model_valid = df_valid[['concatenated','entailment']]
df_model_valid

"""# Testing Pre processing"""

df_test = tfds.as_dataframe(ds_validation)
del df_test['similarity']
df_test['concatenated'] = df_test.apply(lambda x: conc(str(x['text']),str(x['hypothesis'])),axis=1)
df_model_test = df_test[['concatenated','entailment']]
df_model_test

"""# RNN Model"""

# input1 = Input(shape=(1,2))
# rnn_output = SimpleRNN(2,activation='relu')(input1)
# predictions = Activation('softmax')(rnn_output)
# modelo_customizado = Model(inputs=input1,outputs=predictions)

from keras.models import Model
from keras.layers import Input,Dense,SimpleRNN, Activation, Embedding

model = Sequential([
    Embedding(vocab_size,embedding_dim,input_length=max_length),
    SimpleRNN(1),
    Dense(10,activation='relu'),
    Dense(1,activation='sigmoid')
])

model.summary()

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

num_epochs = 30
history = model.fit(padded, train_labels, valid_padded, validation_data=(),epochs=num_epochs)