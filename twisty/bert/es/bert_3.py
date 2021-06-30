import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import re
import unicodedata
import nltk
from nltk.corpus import stopwords
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Input
from tqdm import tqdm
import pickle
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import matplotlib.pyplot as plt
import itertools
from sklearn.utils import shuffle
from tensorflow.keras import regularizers
from transformers import *
from transformers import BertTokenizer, TFBertModel, BertConfig, TFDistilBertModel, DistilBertTokenizer, \
    DistilBertConfig

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate, StratifiedKFold
from sklearn.metrics import confusion_matrix

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def clean_stopwords_shortwords(w):
    stopwords_list = stopwords.words('spanish')
    words = w.split()
    clean_words = [word for word in words if (word not in stopwords_list) and len(word) > 2]
    return " ".join(clean_words)


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = clean_stopwords_shortwords(w)
    w = re.sub(r'@\w+', '', w)
    return w


dataset = pd.read_csv("../../../../../data_set/defesa/t3_defesa_twisty_es.csv", sep=';', encoding='ISO-8859-1')
print(dataset.head())

dbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
dbert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

max_len = 32
sentences = dataset['text']
labels = dataset['ft']
print(len(sentences), len(labels))

dbert_tokenizer.tokenize(sentences[0])

dbert_inp = dbert_tokenizer.encode_plus(sentences[0], add_special_tokens=True, max_length=20, pad_to_max_length=True,
                                        truncation=True)
print(dbert_inp)

id_inp = np.asarray(dbert_inp['input_ids'])
mask_inp = np.asarray(dbert_inp['attention_mask'])
out = dbert_model([id_inp.reshape(1, -1), mask_inp.reshape(1, -1)])
type(out), out
out[0][:, 0, :]

dbert_tokenizer.decode(dbert_inp['input_ids'])


def create_model():
    inps = Input(shape=(max_len,), dtype='int64')
    masks = Input(shape=(max_len,), dtype='int64')
    dbert_layer = dbert_model(inps, attention_mask=masks)[0][:, 0, :]
    dense = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(dbert_layer)
    dropout = Dropout(0.5)(dense)
    pred = Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(0.01))(dropout)
    model = tf.keras.Model(inputs=[inps, masks], outputs=pred)
    print(model.summary())
    return model


model = create_model()

input_ids = []
attention_masks = []

for sent in sentences:
    dbert_inps = dbert_tokenizer.encode_plus(sent, add_special_tokens=True, max_length=max_len, pad_to_max_length=True,
                                             return_attention_mask=True, truncation=True)
    input_ids.append(dbert_inps['input_ids'])
    attention_masks.append(dbert_inps['attention_mask'])

input_ids = np.asarray(input_ids)
attention_masks = np.array(attention_masks)
labels = np.array(labels)

print(len(input_ids), len(attention_masks), len(labels))

print('Preparing the pickle file.....')

pickle_inp_path = './data/dbert_inp.pkl'
pickle_mask_path = './data/dbert_mask.pkl'
pickle_label_path = './data/dbert_label.pkl'

pickle.dump((input_ids), open(pickle_inp_path, 'wb'))
pickle.dump((attention_masks), open(pickle_mask_path, 'wb'))
pickle.dump((labels), open(pickle_label_path, 'wb'))

print('Pickle files saved as ', pickle_inp_path, pickle_mask_path, pickle_label_path)

print('Loading the saved pickle files..')

input_ids=pickle.load(open(pickle_inp_path, 'rb'))
attention_masks=pickle.load(open(pickle_mask_path, 'rb'))
labels=pickle.load(open(pickle_label_path, 'rb'))

print('Input shape {} Attention mask shape {} Input label shape {}'.format(input_ids.shape,attention_masks.shape,labels.shape))

train_inp,val_inp,train_label,val_label,train_mask,val_mask=train_test_split(input_ids,labels,attention_masks,test_size=0.2)

print('Train inp shape {} Val input shape {}\nTrain label shape {} Val label shape {}\nTrain attention mask shape {} Val attention mask shape {}'.format(train_inp.shape,val_inp.shape,train_label.shape,val_label.shape,train_mask.shape,val_mask.shape))

log_dir='dbert_model'
model_save_path='./dbert_model.h5'

callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path,save_weights_only=True,monitor='val_loss',mode='min',save_best_only=True),keras.callbacks.TensorBoard(log_dir=log_dir)]

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)

model.compile(loss=loss,optimizer=optimizer, metrics=[metric])

callbacks= [tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path,save_weights_only=True,monitor='val_loss',mode='min',save_best_only=True),keras.callbacks.TensorBoard(log_dir=log_dir)]
model.compile(loss=loss,optimizer=optimizer, metrics=[metric])
#################################################################

kf = StratifiedKFold(n_splits=10, random_state=123)
pred_test_full =0
cv_f1_score =[]
confusion_matrix_folds = []
i=1

for train_index,test_index in kf.split(input_ids,labels,attention_masks):
    print('{} of KFold {}'.format(i,kf.n_splits))
    input_ids_tr,input_ids_vl = input_ids[train_index],input_ids[test_index]
    labels_tr,labels_vl = labels[train_index],labels[test_index]
    attention_masks_tr, attention_masks_vl = attention_masks[train_index], attention_masks[test_index]

    #model
    fitted = model.fit([input_ids_tr,attention_masks_tr],labels_tr,batch_size=32,epochs=20,callbacks=callbacks)

    trained_model = model
    #trained_model.load_weights(model_save_path)

    preds = trained_model.predict([input_ids_vl, attention_masks_vl], batch_size=32)
    pred_labels = preds.argmax(axis=1)
    f1 = f1_score(labels_vl, pred_labels)
    print(f1)
    cv_f1_score.append(f1)

    conf_mat = confusion_matrix(labels_vl, pred_labels)
    confusion_matrix_folds.append(conf_mat)
    i = i + 1

print(cv_f1_score)
print(confusion_matrix_folds)