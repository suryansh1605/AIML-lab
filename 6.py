from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
doc = pd.read_csv('document.csv', names=['message', 'label'])
print("Total Instances of Dataset: ", doc.shape[0])
doc['labelnum'] = doc.label.map({'pos': 1, 'neg': 0})
X = doc.message
y = doc.labelnum
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)
count_v = CountVectorizer()
Xtrain_dm = count_v.fit_transform(Xtrain)
Xtest_dm = count_v.transform(Xtest)
df = pd.DataFrame(Xtrain_dm.toarray(), columns=count_v.get_feature_names_out())
print(df[0:5])
clf = MultinomialNB()
clf.fit(Xtrain_dm, ytrain)
pred = clf.predict(Xtest_dm)
for doc, p in zip(Xtrain, pred):
 p = 'pos' if p == 1 else 'neg'
 print("%s -> %s" % (doc, p))
print('Accuracy Metrics: \n')
print('Accuracy: ', accuracy_score(ytest, pred))
print('Recall: ', recall_score(ytest, pred))
print('Precision: ', precision_score(ytest, pred))
print('Confusion Matrix: \n', confusion_matrix(ytest, pred))