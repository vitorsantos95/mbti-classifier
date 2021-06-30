import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

dataset_t1 = pd.read_csv("../../../data_set/defesa/t1_defesa.csv", sep='\t')
dataset_t1 = dataset_t1.drop(['Unnamed: 0'], axis=1)
print(dataset_t1.head())

print(dataset_t1.groupby('type_1').count())

print("STARTING TFIDF VECTORIZER...")

vectorizer = TfidfVectorizer(ngram_range=(1, 2), analyzer='word')
x_vectorized_t1 = vectorizer.fit_transform(dataset_t1['comment'])
print(x_vectorized_t1.shape)

X, SEMx_test, Y, SEMy_test = train_test_split(x_vectorized_t1,
                                              dataset_t1['type_1'],
                                              test_size=0.3,
                                              random_state=123,
                                              stratify=dataset_t1['type_1'])

print("MAKING RANGE OF KVALUES...")
i = 30000
kvalues = []
while i > 999:
    kvalues.append(i)
    i -= 1000

print(kvalues)

clf_t1 = LogisticRegression(class_weight='balanced',
                            penalty='l2',
                            dual=False,
                            tol=0.0001,
                            random_state=123,
                            solver='lbfgs',
                            multi_class='ovr',
                            max_iter=150)

print("FIND BEST K-VALUE...")

def findKBest():
    bestScore = 0.0
    bestK = 0
    for kvalue in kvalues:
        best_sel = SelectKBest(chi2, k=kvalue)
        best_fit = best_sel.fit(X, Y)
        x_best = best_fit.transform(X)  # as k melhores features
        #clf = LogisticRegression(class_weight='balanced')
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
print(dataset_t1['type_1'].shape)

import pickle
print("STARTING PICKLE")
model = clf_t1
x_model = x_final
y_model = dataset_t1['type_1']
model.fit(x_model, y_model)
filename = 'logistic_regression_t1_1_2.sav'
pickle.dump(model, open(filename, 'wb'))

print("STARTING LOGISTIC REGRESSION F1")

scores_f1_macro = cross_val_score(clf_t1,
                         x_final,
                         dataset_t1['type_1'],
                         cv=10,
                         scoring='f1_macro',
                         verbose=1)

print(scores_f1_macro)

print("STARTING BASELINE CLASSIFIER F1")

scores_baseline_f1_macro = cross_val_score(DummyClassifier(strategy='stratified', random_state=123, constant=None),
                                 x_final,
                                 dataset_t1['type_1'],
                                 cv=10,
                                 scoring='f1_macro',
                                 verbose=1)

print(scores_baseline_f1_macro)