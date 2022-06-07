import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

dataset = pd.read_csv('./dataset/h1.csv')
x = dataset.iloc[:,1:].values
y = dataset.iloc[:,0].values

ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1,2,3,6,7,8,9,10,11,12,14,15,16])],remainder="passthrough")
x = np.array(ct.fit_transform(x))

le = LabelEncoder()
y = le.fit_transform(y)

X_train,X_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=0)

classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
# classifier = LogisticRegression(random_state=0,solver='lbfgs', max_iter=1000)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

cf = confusion_matrix(y_test,y_pred)

accuracies = cross_val_score(estimator = classifier, X=X_train, y=y_train,cv=10)
print(accuracies.mean())
print(accuracies.std())

filename = 'RFC.sav'
pickle.dump(classifier[0], open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))
res = loaded_model.predict([[0,1,1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,0,0,0,0,1,0,1,1,0,0,1,17,0,20,9]])
print(res)
result = loaded_model.score(X_test, y_test)
print(result)