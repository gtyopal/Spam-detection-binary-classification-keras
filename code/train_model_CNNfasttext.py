# coding:utf-8
from keras.models import Model,Sequential
from keras.layers import Dense, Dropout, Activation,Concatenate, GlobalAveragePooling1D, Embedding, LSTM, GRU,Input,BatchNormalization, Conv1D, GlobalMaxPooling1D
from keras import backend as K
import sys
import time
import random
import numpy as np
from collections import defaultdict
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import data_preprocessing as dp


# parameter set
dataset_path="../data/all_clean.csv"
words_path = "../data/word2id_fasttext.dict"
weights_path = "../data/spam_faxttext.h5"   # model parameters save path
nb_classes = 2  # number of classes
max_len = 1953  # max length of sentence
maxlen = 1953
batch_size = 128  # batch size
nb_epoch = 5  # number of epoch
embedding_length = 300  # length of word embedding
gpu_id = 0  # GPU id for GPU model
words_dict_length = 34400  # number of words in dictionary
max_feature = 34400  # number of words in dictionary
max_features_ngram = 481918
words_dict = {}  # dictionary
model_type = 'CNNFastText'  # support textcnn, textrnn, attnlstm, fasttext


# Word dictionary prepare and data split
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

# model training
df_dataset = pd.read_table(dataset_path,encoding='latin-1',sep="\t",low_memory=False)
print (type(df_dataset))
print("loading dataset...")
word2id = generate_words(df_dataset)
words_dict_length = len(word2id)
print(word2id)
print(words_dict_length)

def save_words(word2id, words_path):
    """ save words and id """
    with open(words_path, "w") as fw:
        for w in word2id:
            fw.write(w + "\t" + str(word2id[w]) + "\n")

print("save words...")
save_words(word2id, words_path)


def load_words(words_path):
    """ load words and id """
    word2id = {}
    with open(words_path, "r") as fr:
        for line in fr.readlines():
            line = line.strip().split("\t")
            word2id[line[0]] = int(line[1])
    return word2id

def split_train_test(dataset_path, split_rate = 0.1):
    """ generate train and test dataset, then map words to index """
    word2id=load_words(words_path)
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
    return x_train, y_train, x_test, y_test, len(word2id), max_len_sent


print("split train and test...")
x_train, y_train, x_test, y_test, max_features, max_len = split_train_test(dataset_path)
print("max_features:",max_features)
print("Max length of sentence", max_len)

# # FastText Data Processing by adding n-grams:
def create_ngram_set(input_list, ngram_value=2):
        return set(zip(*[input_list[i:] for i in range(ngram_value)]))

def add_ngram(sequences, token_indice, ngram_range=2):
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for ngram_value in range(2, ngram_range + 1):
            for i in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)
    return new_sequences

def FastText_data(x_train, x_test, max_features, ngram_range=2):
    if ngram_range > 1:
        ngram_set = set()
        for input_list in x_train:
            for i in range(2, ngram_range + 1):
                set_of_ngram = create_ngram_set(input_list, ngram_value=i)
                ngram_set.update(set_of_ngram)
        start_index = max_features + 1
        token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
        indice_token = {token_indice[k]: k for k in token_indice}
        try:
            max_features = np.max(list(indice_token.keys()))
        except ValueError:
            pass
        x_train = add_ngram(x_train, token_indice, ngram_range)
        x_test = add_ngram(x_test, token_indice, ngram_range)
    return x_train, x_test, max_features

### FastText
x_train, x_test, max_features_ngram = FastText_data(x_train, x_test, max_features, ngram_range=2)
print("x_train:", np.shape(x_train))
print("x_test:", np.shape(x_test))
print("y_train:", np.shape(y_train))
print("y_test:", np.shape(y_test))
print("max_features_ngram:",max_features_ngram)
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


# # FastText Model
def CNN_FastText_model(max_features_ngram=481918, maxlen=1953, embedding_length=300, nb_classes=2):
    """ CNNFastText """
    max_features_ngram = max_features_ngram + 1 # input dims
    inputs = Input(shape=(maxlen,))
    embed = Embedding(max_features_ngram, embedding_length)(inputs)
    conv_3 = Conv1D(filters=256, kernel_size=3, padding="valid", activation="relu", strides=1)(embed)
    conv_4 = Conv1D(filters=256, kernel_size=4, padding="valid", activation="relu", strides=1)(embed)
    conv_5 = Conv1D(filters=256, kernel_size=5, padding="valid", activation="relu", strides=1)(embed)
    pool_3 = GlobalMaxPooling1D()(conv_3)
    pool_4 = GlobalMaxPooling1D()(conv_4)
    pool_5 = GlobalMaxPooling1D()(conv_5)
    cat = Concatenate()([pool_3, pool_4, pool_5])
    output = Dropout(0.25)(cat)
    dense1 = Dense(256, activation='relu')(output)
    bn = BatchNormalization()(dense1)
    dense2 = Dense(nb_classes, activation='softmax')(bn)
    model = Model(inputs=inputs, outputs=dense2)
    return model


model = CNN_FastText_model(max_features_ngram, maxlen, embedding_length, nb_classes)
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ["accuracy"])
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
                    callbacks=[checkpoint])
toc = time.process_time()
print("training ended...")
print (" ----- total Computation time = " + str((toc - tic)/3600) + " hours ------ ")
print("model summary")
model.summary()


def load_model(model_type):
    """ load model """
    word2id = load_words(words_path)
    max_features = len(word2id)
    if model_type.lower() == "cnnfasttext":
        model = CNN_FastText_model(max_features_ngram, maxlen, embedding_length, nb_classes)
    model.load_weights(weights_path)
    return model, word2id

model, word2id = load_model("CNNFastText")

def evaluation(self, model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_pred = [int(i[1] + 0.5) for i in y_pred]
    target_names = ['class ' + str(i) for i in range(0, self.nb_classes)]
    y_test = y_test.tolist()
    y_test = [i.index(1.0) for i in y_test]
    print(classification_report(y_test, y_pred, target_names=target_names))
    print('\n')
    print(confusion_matrix(y_test, y_pred))



def predict(text, model, word2id):
    """ predict """
    text = dp.denoise_text(text)
    if not text.strip():
        return [[1.0, 0.0]]
    x_test = [word2id.get(w, 0) for w in text.split()]
    x_test = sequence.pad_sequences([x_test], maxlen=maxlen)
    y_predicted = model.predict(x_test, batch_size=1)
    return y_predicted


s = "Iphone to iCloud movies not playing on mac"
model, word2id = load_model("CNNFastText")
print("Iphone to iCloud movies not playing on mac")
print(predict(s, model, word2id))
print("ham prob", "spam prob")


s = "I post a love story on facebook"
model, word2id = load_model("CNNFastText")
print("I post a love story on facebook")
print(predict(s, model, word2id))
print("ham prob", "spam prob")




