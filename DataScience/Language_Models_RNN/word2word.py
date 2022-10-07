from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense,LSTM,Embedding,Activation,Flatten,Dropout,Bidirectional
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

### WORD2WORD
corpus = "Quero jogar futebol hoje \n Hoje não tem futebol"
corpus = corpus.lower()

vocab = {}
tokens = []

# creating vocabulary and tokens
for sentence in corpus.split("\n"):
    for word in sentence.split():
        tokens.append(word)
        if word in vocab:
            vocab[word] += 1
        else:
            vocab[word] = 1
vocab_size = len(vocab)


...

# creating word_index
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(vocab)


word2index = tokenizer.word_index
word2index['<OOV>'] = 0
index2word = {}

for key in word2index:
    value = word2index[key]
    index2word[value] = key


# está usando tokens
# objetivo: prever a proxima palavra da sen

# o valor 0 do x_train serve para sinalizar o inicio da predição
# o valor 0 em y_train serve para sinalizar que não existe mais palavras para prever
X_train = [0]
y_train = [word2index[tokens[0]]]
for i in range(0,len(tokens)-1):
    X_train.append(word2index[tokens[i]])
    y_train.append(word2index[tokens[i+1]])
X_train.append(word2index[tokens[len(tokens)-1]])
y_train.append(0)

# Creating the model
model = Sequential()
model.add(Embedding(input_dim=vocab_size+1,output_dim=32,input_length=1))
model.add(Bidirectional(LSTM(256,activation='relu')))
model.add(Dropout(0.5))
model.add(Dense(vocab_size+1,activation='softmax'))
...

# Training the model
sgd = SGD(learning_rate = 0.001)
model.compile(optimizer=sgd,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history = model.fit(X_train,y_train,batch_size=16,epochs=3)


import numpy as np
frase = 'futebol quero jogar'

for w in frase.split():
    idx = word2index[w]
    prob = model.predict([idx])
    pal = np.argmax(prob)
    print(f'Palavra atual: {index2word[idx]} Proxima palavra: {index2word[pal]}')


### SENTENCE2WORD

novel_corpus = []
y_train = []
for sentence in corpus.split('\n'):

    novos_termos = sentence.split()

    for i in range(0,len(novos_termos)):
        lista = novos_termos[:i+1]
        novel_corpus.append(lista)
        if i < len(novos_termos)-1:
            y_train.append(word2index[novos_termos[i+1]])
        else:
            y_train.append(0)
max_length = max([len(sentence) for sentence in novel_corpus])
train_sequences = tokenizer.texts_to_sequences(novel_corpus)
trunc_type = 'post'
padding_type = 'pre'

train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

y_train = to_categorical(y_train)

model2 = Sequential()
model2.add(Embedding(input_dim=vocab_size+1,output_dim=32,input_length=max_length))
model2.add(Bidirectional(LSTM(256,activation='relu')))
model2.add(Dropout(0.5))
model2.add(Dense(vocab_size+1,activation='softmax'))

sgd = SGD(learning_rate = 0.001)
model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(train_padded,y_train,batch_size=16,epochs=3)

### Modelo de Linguagem com Corpus da Reuters

from keras.datasets import reuters

vocab_size = 3000

(x_train,y_train_int),(x_test2,y_test2) = reuters.load_data(num_words=vocab_size,test_split=0.3)
word2index = reuters.get_word_index()

index2word = {}

for key,value in word2index.items():
  index2word[value] = key

print(' '.join([index2word[x] for x in x_train[0]]))


