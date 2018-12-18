# coding:utf-8
from keras.models import Model,Sequential
from keras.layers import Dense, Dropout, Activation,Concatenate, Embedding, LSTM, GRU,Input,BatchNormalization, Conv1D, GlobalMaxPooling1D
from keras import backend as K
from attention_layer import Attention_layer
import sys
import time
import random
import numpy as np
from collections import defaultdict
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import Model,Sequential
from keras.layers import Dense, Dropout, Activation,Concatenate, Embedding,LSTM, GRU,Input,BatchNormalization,Conv1D,GlobalMaxPooling1D
from keras import backend as K
import pandas as pd
import sklearn
from sklearn import metrics
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split, StratifiedKFold
import data_preprocessing


def generate_words(df_dataset):
    """ generate words dict """
    words_dict = defaultdict(int)
    for item in df_dataset["message"]:
        if not isinstance(item, str):
            continue
        for word in item.split():
            words_dict[word] += 1
    count_sort = sorted(words_dict.items(), key=lambda e:-e[1])
    word2id, idx = {'pad': 0}, 1
    for w in count_sort:
        if w[1] > 2:
            word2id[w[0]] = idx
            idx += 1
    return word2id


def save_words(word2id, words_path):
    """ save words and id """
    with open(words_path, "w") as fw:
        for w in word2id:
            fw.write(w + "\t" + str(word2id[w]) + "\n")


def load_words(words_path):
    """ load words and id """
    word2id = {}
    with open(words_path, "r") as fr:
        for line in fr.readlines():
            line = line.strip().split("\t")
            word2id[line[0]] = int(line[1])
    return word2id



def split_train_test(dataset_path, word2id, split_rate = 0.1):
    """ generate train and test dataset, then map words to index """
    df_dataset = pd.read_csv(dataset_path,encoding = 'latin-1',sep = "\t")
    item2id = []  # [(label, message2id)]
    ### Add
    max_len_sent = 0
    for i in df_dataset.index:
        label, message = df_dataset.loc[i].values[0], df_dataset.loc[i].values[1]
        if not isinstance(message, str):
            continue
        message2id = [word2id.get(w, 0) for w in message.split()]
        item2id.append((label, message2id))
        ### Add
        if len(message2id) > max_len_sent:
            max_len_sent = len(message2id)
    random.shuffle(item2id)
    x_train, y_train, x_test, y_test = [], [], [], []
    for item in item2id[:int(len(item2id)*split_rate)]:
        x_test.append(item[1])
        y_test.append(item[0])
    for item in item2id[int(len(item2id)*split_rate):]:
        x_train.append(item[1])
        y_train.append(item[0])
    ### Add
    print("Max features: ", len(word2id))
    print("Max length of sentence", max_len_sent)
    return x_train, y_train, x_test, y_test, len(word2id), max_len_sent


def AttnGRU(words_dict_length=35740, max_len=2170, embedding_length=300, nb_classes=2):
    """ Attention Model with GRU model"""
    max_features = words_dict_length + 1 # input dims
    inputs = Input(shape=(maxlen,))
    embed = Embedding(max_features, embedding_length)(inputs)
    gru = GRU(256, dropout=0.25, recurrent_dropout=0.25, return_sequences=True)(embed)
    output = Attention_layer()(gru)
    dense1 = Dense(256, activation='relu')(output)
    bn = BatchNormalization()(dense1)
    dense2 = Dense(nb_classes, activation='softmax')(bn)
    model = Model(inputs=inputs, outputs=dense2)
    return model


# define tensorboard
tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)



dataset_path="./data/all_clean.csv"
words_path = "./data/word2id.dict"
weights_path = "./data/spam_attngru.h5"   # model parameters save path
nb_classes = 2  # number of classes
maxlen = 2170  # max length of sentence
batch_size = 128  # batch size
nb_epoch = 100  # number of epoch
embedding_length = 300  # length of word embedding 
model_type = 'AttnGRU'  # support textcnn, textrnn, attnlstm, 
gpu_id = 0  # GPU id for GPU model
words_dict_length = 35740  # number of words in dictionary
words_dict = {}  # dictionary


# In[13]:


df_dataset = pd.read_csv(dataset_path,encoding='latin-1',sep="\t")
print("loading dataset...")
word2id = generate_words(df_dataset)
words_dict_length = len(word2id)
print("save words...")
save_words(word2id, words_path)
print("split train and test...")
x_train, y_train, x_test, y_test, max_features, max_len = split_train_test(dataset_path, word2id)
print("padding")
x_train = sequence.pad_sequences(np.asarray(x_train), maxlen=max_len)
x_test = sequence.pad_sequences(np.asarray(x_test), maxlen=max_len)
y_train = np_utils.to_categorical(np.asarray(y_train))
y_test = np_utils.to_categorical(np.asarray(y_test))
print("init model")
model_type = model_type.lower()
if model_type.lower() == "textcnn":
    model = TextCNN(words_dict_length, max_len, embedding_length, nb_classes)
elif model_type.lower() == "textrnn":
    model = TextRNN(words_dict_length, max_len, embedding_length, nb_classes)
elif model_type.lower() == "attngru":
    model = AttnGRU(words_dict_length, max_len, embedding_length, nb_classes)
elif model_type.lower() == "attnlstm":
    model = AttnLSTM(words_dict_length, max_len, embedding_length, nb_classes)
elif model_type.lower() == 'textcnn_embed':
    embedding_matrix = create_embedding(embedding_file, word2id, embedding_length)
    model = TextCNN_with_PreEmbedding(embedding_matrix, words_dict_length, max_len, embedding_length, nb_classes)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
print("model summary")
model.summary()
print("checkpoint_dir: %s"%weights_path)
checkpoint = ModelCheckpoint(weights_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')



print("training started...")
tic = time.process_time()
history = model.fit(x_train,
          y_train,
          batch_size=batch_size,
          epochs=nb_epoch,
          validation_data=(x_test, y_test),
          shuffle=1,
          callbacks=[checkpoint,tensorboard]).history
toc = time.process_time()
print("training ended...")
print (" -----total Computation time = " + str((toc - tic)/3600) + " hours------")




def load_model(model_type):
    """ load model """
    word2id = load_words(words_path)
    max_features = len(word2id)
    model_type = model_type.lower()
    if model_type.lower() == "textcnn":
        model = TextCNN(max_features, maxlen, embedding_length, nb_classes)
    elif model_type.lower() == "textrnn":
        model = TextRNN(max_features, maxlen, embedding_length, nb_classes)
    elif model_type.lower() == "attngru":
        model = AttnGRU(max_features, maxlen, embedding_length, nb_classes)
    elif model_type.lower() == "attnlstm":
        model = AttnLSTM(max_features, maxlen, embedding_length, nb_classes)
    # model = AttnLSTM(max_features, maxlen, embedding_length, nb_classes)
    model.load_weights(weights_path)
    return model, word2id


model, word2id = load_model("AttnGRU")



def predict(text, model, word2id):
    """ predict """
    text = data_preprocessing.denoise_text(text)
    if not text.strip():
        return [[1.0, 0.0]]
    x_test = [word2id.get(w, 0) for w in text.split()]
    x_test = sequence.pad_sequences([x_test], maxlen=maxlen)
    y_predicted = model.predict(x_test, batch_size=1)
    return y_predicted


s = "I post a  story on facebook"
model, word2id = load_model("AttnGRU")
print(predict(s, model, word2id))
print("ham prob", "spam prob")

