# Save Model Using Pickle
from sklearn.datasets import fetch_openml
import pickle
mnist = fetch_openml("mnist_784")
X,y = mnist["data"],mnist["target"]
X_train, X_test = X[:60000],X[60000:]
y_train,y_test = y[:60000],y[60000:]
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(tol = 0.1)
clf.fit(X_train,y_train)

# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))

# some time later...

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)
