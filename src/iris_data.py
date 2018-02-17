import os

import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#load dataset
current_file_path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(current_file_path, "../datasets/train.csv")
fd_dataset = open(path,"r")
names = ['buying_price', 'maintainence_cost', 'number_of_doors', 'number_of_seats', 'luggage_boot_size', 'safety_rating','popularity']
dataset = pandas.read_csv(fd_dataset,names=names)

array = dataset.values
X = array[1:,0:6]
Y = array[1:,6]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                random_state=seed)

seed = 7
scoring = 'accuracy'

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)




# Make predictions on validation dataset
knn = DecisionTreeClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# Make prediction on test dataset
names = names.pop()
fd_test = open(os.path.join(current_file_path, "../datasets/test.csv"),"r")
test_dataset = pandas.read_csv(fd_test,header=None)
test_array =test_dataset.values
X_test =test_array[:,0:6]
predictions = knn.predict(X_test)

print predictions