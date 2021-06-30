import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.dummy import DummyClassifier

dataset = pd.read_csv("../../../../data_set/defesa/tweets_it.csv", sep=';', encoding='ISO-8859-1')
print(dataset.head())

x_text = dataset["text"]
y_pj = dataset["jp"]

vectorizer = TfidfVectorizer(ngram_range=(1, 1), analyzer='word')
x_vectorized_t1 = vectorizer.fit_transform(x_text)
print(x_vectorized_t1.shape)

# X, SEMx_test, Y, SEMy_test = train_test_split(x_vectorized_t1,
#                                               x_text,
#                                               test_size=0.3,
#                                               random_state=123,
#                                               stratify=y_pj)

X = x_vectorized_t1
Y = dataset["jp"]

print("MAKING RANGE OF KVALUES...")
i = 30000
kvalues = []
while i > 999:
    kvalues.append(i)
    i -= 1000

print(kvalues)

clf_t1 = DummyClassifier(strategy='most_frequent', random_state=123, constant=None)

print("FIND BEST K-VALUE...")

def findKBest():
    bestScore = 0.0
    bestK = 0
    for kvalue in kvalues:
        best_sel = SelectKBest(chi2, k=kvalue)
        best_fit = best_sel.fit(X, Y)
        x_best = best_fit.transform(X)  # as k melhores features
        mean_score = float("{:2.2f}".format(cross_val_score(clf_t1, x_best, Y, scoring='f1_macro', cv=5).mean()))
        if mean_score >= bestScore:
            bestScore = mean_score
            bestK = kvalue
    return bestK

k = findKBest()

print("o melhor valor de k foi " + str(k))
best_sel = SelectKBest(chi2, k=k)
best_fit = best_sel.fit(X, Y)   # treina apenas na porcao reduzida dos dados
x_final = best_fit.transform(x_vectorized_t1)   # aplica a reducao de bestK ao conjunto completo

print(x_final.shape)
print(x_text.shape)

print("STARTING PICKLE")
model = clf_t1
x_model = x_final
y_model = y_pj
model.fit(x_model, y_model)
filename = 'baseline_t4_1_1.sav'
pickle.dump(model, open(filename, 'wb'))

print("STARTING BASELINE F1")
scores_f1_macro = cross_val_score(clf_t1,
                         x_final,
                         y_pj,
                         cv=10,
                         scoring='f1_macro',
                         verbose=1)

print(scores_f1_macro)