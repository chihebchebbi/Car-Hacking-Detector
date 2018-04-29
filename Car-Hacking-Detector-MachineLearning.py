
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split


Features = pd.read_csv("Features1.csv")
data = pd.read_csv("dos1.csv")
Res = data.loc[:,"R"]
Res1 = []
Res1 = Res.replace("R","0")
Res2 = []
Res2 = Res1.replace("T","1")
flags = []
flags = Res2
Flags = flags.to_frame()
fls = Flags[:3634582]
X_train, X_test, y_train, y_test = train_test_split(Features, fls, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, np.ravel(y_train,order='C'))
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
