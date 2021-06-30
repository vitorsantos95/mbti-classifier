## for data
import json
import pandas as pd
import numpy as np
## for plotting
import matplotlib.pyplot as plt
import seaborn as sns
## for processing
import re
import nltk
## for bag-of-words
from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing
## for word embedding
import gensim
import gensim.downloader as gensim_api
## for deep learning
from tensorflow.keras import models, layers, preprocessing as kprocessing
from tensorflow.keras import backend as K

from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate
from sklearn.metrics import confusion_matrix

## for bert language model
##import transformers

dataset = pd.read_csv("../../../../data_set/defesa/t1_defesa_twisty_de.csv", sep=';', encoding='ISO-8859-1')
print(dataset.head())

y_train = dataset['ei']

def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
    ## clean (convert to lowercase and remove punctuations and characters and then strip)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    ## Tokenize (convert from string to list)
    lst_text = text.split()
    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in lst_stopwords]
    ## Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]

    ## Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]

    ## back to string from list
    text = " ".join(lst_text)
    return text

lst_stopwords = nltk.corpus.stopwords.words("german")

dataset["text_clean"] = dataset["text"].apply(lambda x:
          utils_preprocess_text(x, flg_stemm=False, flg_lemm=True,
          lst_stopwords=lst_stopwords))

print(dataset.head())

corpus = dataset["text_clean"]

## create list of lists of unigrams
lst_corpus = []
for string in corpus:
    lst_words = string.split()
    lst_grams = [" ".join(lst_words[i:i + 1])
                 for i in range(0, len(lst_words), 1)]
    lst_corpus.append(lst_grams)

## detect bigrams and trigrams
bigrams_detector = gensim.models.phrases.Phrases(lst_corpus,
                                                 delimiter=" ".encode(),
                                                 min_count=5,
                                                 threshold=10)
bigrams_detector = gensim.models.phrases.Phraser(bigrams_detector)
trigrams_detector = gensim.models.phrases.Phrases(bigrams_detector[lst_corpus],
                                                  delimiter=" ".encode(),
                                                  min_count=5,
                                                  threshold=10)
trigrams_detector = gensim.models.phrases.Phraser(trigrams_detector)

embedding_dim = 300

## fit w2v
nlp = gensim.models.word2vec.Word2Vec(lst_corpus,
                                      size=embedding_dim,
                                      window=8,
                                      min_count=1,
                                      sg=1,
                                      iter=30)

nlp.wv.save_word2vec_format('lstm_1.bin', binary=True)

word = "data"
nlp[word].shape

## tokenize text
tokenizer = kprocessing.text.Tokenizer(lower=True, split=' ',
                                       oov_token="NaN",
                                       filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(lst_corpus)
dic_vocabulary = tokenizer.word_index

## create sequence
lst_text2seq = tokenizer.texts_to_sequences(lst_corpus)

## padding sequence
X_train = kprocessing.sequence.pad_sequences(lst_text2seq,
                                             maxlen=15,
                                             padding="post",
                                             truncating="post")

sns.heatmap(X_train==0, vmin=0, vmax=1, cbar=False)
plt.show()

i = 0

## list of text: ["I like this", ...]
len_txt = len(dataset["text_clean"].iloc[i].split())
print("from: ", dataset["text_clean"].iloc[i], "| len:", len_txt)

## sequence of token ids: [[1, 2, 3], ...]
len_tokens = len(X_train[i])
print("to: ", X_train[i], "| len:", len(X_train[i]))

## vocabulary: {"I":1, "like":2, "this":3, ...}
print("check: ", dataset["text_clean"].iloc[i].split()[0],
      " -- idx in vocabulary -->",
      dic_vocabulary[dataset["text_clean"].iloc[i].split()[0]])
print("vocabulary: ", dict(list(dic_vocabulary.items())[0:5]), "... (padding element, 0)")

## start the matrix (length of vocabulary x vector size) with all 0s
embeddings = np.zeros((len(dic_vocabulary) + 1, 300))

for word, idx in dic_vocabulary.items():
    ## update the row with vector
    try:
        embeddings[idx] = nlp[word]
    ## if word not in model then skip and the row stays all 0s
    except:
        pass

word = "data"
print("dic[word]:", dic_vocabulary[word], "|idx")
print("embeddings[idx]:", embeddings[dic_vocabulary[word]].shape, "|vector")

## code attention layer
def attention_layer(inputs, neurons):
    x = layers.Permute((2,1))(inputs)
    x = layers.Dense(neurons, activation="softmax")(x)
    x = layers.Permute((2,1), name="attention")(x)
    x = layers.multiply([inputs, x])
    return x

## input
x_in = layers.Input(shape=(15,))

## embedding
x = layers.Embedding(input_dim=embeddings.shape[0],
                     output_dim=embeddings.shape[1],
                     weights=[embeddings],
                     input_length=15, trainable=False)(x_in)

## apply attention
x = attention_layer(x, neurons=15)

## 2 layers of bidirectional lstm
x = layers.Bidirectional(layers.LSTM(units=15, dropout=0.2, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(units=15, dropout=0.2))(x)

## final dense layers
x = layers.Dense(64, activation='relu')(x)
y_out = layers.Dense(3, activation='softmax')(x)

## compile
def buildModel():
    model = models.Model(x_in, y_out)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

## encode y
dic_y_mapping = {n:label for n,label in enumerate(np.unique(y_train))}
inverse_dic = {v:k for k,v in dic_y_mapping.items()}
y_train = np.array([inverse_dic[y] for y in y_train])

# ## train
# training = model.fit(x=X_train, y=y_train, batch_size=256,
#                      epochs=10, shuffle=True, verbose=0,
#                      validation_split=0.3)
#
# #estimator.fit(X, labels)

estimator = KerasClassifier(build_fn=buildModel,
                            batch_size=512,
                            epochs=100,
                            verbose=1)

model_bkp = buildModel()
model_bkp.save('lstm_1_model.h5')
model_bkp.save_weights('lstm_1_weights.h5')

f1_score = cross_val_score(estimator,
                        X_train,
                        y_train,
                        cv=10,
                        scoring='f1_macro',
                        verbose=1)


y_pred = cross_val_predict(estimator, X_train, y_train, cv=10)
conf_mat = confusion_matrix(y_train, y_pred)

print(f1_score)

print(conf_mat)