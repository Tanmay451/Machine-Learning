# All the python libraries used

import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt

# Quandl

df = quandl.get("WIKI/GOOGL")
print(df.head(3))

# In[ ]:

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
print(df.head(3))

# In[ ]:

df['HL. PCT'] = (df['Adj. High'] - df['Adj. Close'])/df['Adj. Close']* 100.0
df['PIC_change'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open']* 100.0
df = df[['Adj. Close','HL. PCT','PIC_change','Adj. Volume']] 

# In[ ]:

print(df.head(3))

# In[ ]:

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace = True)
forecast_out = int(math.ceil(0.01*len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace = True)

# In[ ]:

print(df.head(3))

# In[ ]:

X = np.array(df.drop(['label'],1))
y = np.array(df['label'])
X = preprocessing.scale(X)
y = np.array(df['label'])

# For random split of data

# In[ ]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
clf = LinearRegression()

# to make training faster just LinearRegression(n_jods=10)

clf.fit(X_train, y_train)
accuracy = clf.score(X_test,y_test)
print(accuracy)

# In[ ]:

#another algo for classification but it is not so accurate "support vector regration'
clf = svm.SVR()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test,y_test)
print(accuracy)

# In[ ]:

# if we want to save our classifier to avoid procassing time every time we use it
# we should do it after training with data
import pickle
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
with open('linearRegration.pickle','wb') as f:
    pickle.dump(clf,f)
#till now we have load our data and now we will reading it......... Once we train we can comment out above code or the training part and just use the file instade
pickle_in = open('linearRegration.pickle','rb')
clf = pickle.load(pickle_in)

# In[ ]:

X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)

# In[ ]:

print(last_date.head(3))

# In[ ]:

forecast_set = clf.predict(X_lately)
df['Forecast'] = np.nan

last_date = df.iloc[:0:-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# Plotting a graph

# In[8]:

xs = [1,2,3,4,5,6]
ys = [5,4,3,2,6,7]
ya = [3,5,5,6,8,9]
y1 = plt.plot(xs,ys) # plot for line
y2 = plt.scatter(xs,ya) #marking a point
plt.show(y1)
plt.show(y2)
