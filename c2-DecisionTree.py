import csv
import numpy as np

with open('./titanic.csv', 'rb') as csvfile:
    titanic_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    row = titanic_reader.next()
    feature_names = np.array(row)

    titanic_X, titanic_y = [], []
    for row in titanic_reader:
        titanic_X.append(row)
        titanic_y.append(row[2])

    titanic_X = np.array(titanic_X)
    titanic_y = np.array(titanic_y)


titanic_X = titanic_X[:, [1, 4, 10]]
feature_names = feature_names[[1, 4, 10]]

print feature_names
print titanic_X[12], titanic_y[12]

ages = titanic_X[:, 1]
mean_age = np.mean(titanic_X[ages != 'NA', 1].astype(np.float))
titanic_X[titanic_X[:, 1] == 'NA', 1] = mean_age

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
enc = LabelEncoder()
label_encoder = enc.fit(titanic_X[:, 2])
print "Categorical classes:", label_encoder.classes_

integer_classes = label_encoder.transform(label_encoder.classes_)
print "Integer classes:", integer_classes

#print titanic_X[0]
titanic_X[:, 2] = label_encoder.transform(titanic_X[:, 2])
print titanic_X[12], titanic_y[12]

label_encoder = enc.fit(titanic_X[:, 0])
print "Categorical classes:", label_encoder.classes_
integer_classes = label_encoder.transform(label_encoder.classes_).reshape(3, 1)
print "Integer classes:", integer_classes

enc = OneHotEncoder()
one_hot_encoder = enc.fit(integer_classes)
num_of_rows = titanic_X.shape[0]

t = label_encoder.transform(titanic_X[:, 0]).reshape(num_of_rows, 1)
new_features = one_hot_encoder.transform(t)
titanic_X = np.concatenate([titanic_X, new_features.toarray()], axis=1)
titanic_X = np.delete(titanic_X, [0], 1)
feature_names = ['age', 'sex', 'first_class', 'second_class', 'third_class']
titanic_X = titanic_X.astype(float)
titanic_y = titanic_y.astype(float)

print titanic_X[0], titanic_y[0]

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(titanic_X, titanic_y, test_size=0.25, random_state=33)

from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=5)
clf = clf.fit(X_train, y_train)

from sklearn import metrics
def measure_performance(X, y, clf, show_accuracy=True, show_classification_report=True, show_confussion_matrix=True):
    y_pred=clf.predict(X)
    if show_accuracy:
        print "Accuracy:{0:.3f}".format(
            metrics.accuracy_score(y, y_pred)
        ), "\n"

    if show_classification_report:
        print "Classification report"
        print metrics.classification_report(y, y_pred), "\n"

    if show_confussion_matrix:
        print "Confussion matrix"
        print metrics.confusion_matrix(y, y_pred), "\n"

measure_performance(X_train, y_train, clf)

from sklearn.cross_validation import cross_val_score, LeaveOneOut
from scipy.stats import sem

def loo_cv(X_train, y_train, clf):
    loo = LeaveOneOut(X_train[:].shape[0])
    scores = np.zeros(X_train[:].shape[0])
    for train_index, test_index in loo:
        X_train_cv, X_test_cv = X_train[train_index], X_train[test_index]
        y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]
        clf = clf.fit(X_train_cv, y_train_cv)
        y_pred = clf.predict(X_test_cv)
        scores[test_index] = metrics.accuracy_score(y_test_cv.astype(int), y_pred.astype(int))
    print ("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores))


loo_cv(X_train, y_train, clf)


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10, random_state=33)
clf = clf.fit(X_train, y_train)
loo_cv(X_train, y_train, clf)
