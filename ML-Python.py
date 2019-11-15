#!/usr/bin/env python
# coding: utf-8

# # All the python libraries used

# In[2]:


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
import random


# ### Quandl

# In[ ]:


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


# ### For random split of data

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


# ### Plotting a graph

# In[ ]:


xs = np.array([1,2,3,4,5,6], dtype = np.float64)
ys = np.array([5,4,6,5,6,7], dtype = np.float64)
ya = [3,5,5,6,8,9]
l1 = plt.scatter(xs,ys) # plot for line
#l2 = plt.plot(xs,ya) #marking a point

def best_fit_line_and_intersept(xs,ys):
    m= ((mean(xs)*mean(ys)-mean(xs*ys))/
        ((mean(xs)*mean(xs)) - mean(xs*xs)))
    b = mean(ys) - m*mean(xs)
    return m , b
def squared_error(ys_orig, ys_line):
    return sum((ys_line-ys_orig)**2)

def coefficient_of_detetmination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1-(squared_error_regr/squared_error_y_mean)
m , b = best_fit_line_and_intersept(xs,ys)


regration_line = []
for x in xs:
    regration_line.append((m*x+b))

r_squared = coefficient_of_detetmination(ys, regration_line)
print(r_squared)
final_line = plt.plot(xs,regration_line)
plt.show()


# ### K Nearest Neighbors

# In[ ]:


from scipy.spatial import distance
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

#########################CLASSIFIER##############################
def euc(a,b):
    return distance.euclidean(a,b)


class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions
    
    def closest(self, row):
        best_dist = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if (dist < best_dist):
                best_dist = dist
                best_index = i
        return self.y_train[best_index]
        
##########################CLASSIFIER##############################

my_classifier = ScrappyKNN()
my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)


print (accuracy_score(y_test, predictions))


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from math import sqrt
from collections import Counter
style.use('fivethirtyeight')
dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]
[[plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0], new_features[1], s=100)

plt.show()
for i in dataset:
    for ii in dataset[i]:
        plt.scatter(ii[0],ii[1],s=100,color=i)
def k_nearest_neighbors(data, predict, k=3):
    return vote_result
def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')

    return vote_result


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from math import sqrt
from collections import Counter
style.use('fivethirtyeight')

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
        
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance,group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]
[[plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]] for i in dataset]
# same as:
##for i in dataset:
##    for ii in dataset[i]:
##        plt.scatter(ii[0],ii[1],s=100,color=i)
        
plt.scatter(new_features[0], new_features[1], s=100)

result = k_nearest_neighbors(dataset, new_features)
plt.scatter(new_features[0], new_features[1], s=100, color = result)  
plt.show()


# ### SVM

# In[5]:


import numpy as np
from sklearn import preprocessing, neighbors, svm
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('https://raw.githubusercontent.com/Tanmay451/Machine-Learning/master/breast-cancer-wisconsin.data.txt?token=AJO6DFD6IHPO6ZUQAYETARC5IWAHY')
df.replace('?',-99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = svm.SVC()

clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print(confidence)

example_measures = np.array([[4,2,1,1,1,2,3,2,1]])
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)
print(prediction)


# In[11]:


import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

class Support_Vector_Machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1:'r',-1:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
    # train
    def fit(self, data):
        self.data = data
        # { ||w||: [w,b] }
        opt_dict = {}

        transforms = [[1,1],
                      [-1,1],
                      [-1,-1],
                      [1,-1]]

        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)
                  

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
    
        all_data = None
        # support vectors yi(xi.w+b) = 1
        

        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      # point of expense:
                      self.max_feature_value * 0.001,
                      ]

        
        
        # extremely expensive
        b_range_multiple = 2
        # we dont need to take as small of steps
        # with b as we do w
        b_multiple = 5
        latest_optimum = self.max_feature_value*10
        
        for step in step_sizes:
            w = np.array([latest_optimum,latest_optimum])
            # we can do this because convex
            optimized = False
            while not optimized:
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple),
                                   self.max_feature_value*b_range_multiple,
                                   step*b_multiple):
                    for transformation in transforms:
                        w_t = w*transformation
                        found_option = True
                        # weakest link in the SVM fundamentally
                        # SMO attempts to fix this a bit
                        # yi(xi.w+b) >= 1
                        # 
                        # #### add a break here later..
                        for i in self.data:
                            for xi in self.data[i]:
                                yi=i
                                if not yi*(np.dot(w_t,xi)+b) >= 1:
                                    found_option = False
                                    #print(xi,':',yi*(np.dot(w_t,xi)+b))
                                    
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t,b]

                if w[0] < 0:
                    optimized = True
                    print('Optimized a step.')
                else:
                    w = w - step

            norms = sorted([n for n in opt_dict])
            #||w|| : [w,b]
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0]+step*2
            
        for i in self.data:
            for xi in self.data[i]:
                yi=i
                print(xi,':',yi*(np.dot(self.w,xi)+self.b)) 
    def predict(self,features):
            # sign( x.w+b )
            classification = np.sign(np.dot(np.array(features),self.w)+self.b)
            if classification !=0 and self.visualization:
                self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])
            return classification

    def visualize(self):
        [[self.ax.scatter(x[0],x[1],s=100,color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        # hyperplane = x.w+b
        # v = x.w+b
        # psv = 1
        # nsv = -1
        # dec = 0
        def hyperplane(x,w,b,v):
            return (-w[0]*x-b+v) / w[1]

        datarange = (self.min_feature_value*0.9,self.max_feature_value*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # (w.x+b) = 1
        # positive support vector hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2], 'k')

        # (w.x+b) = -1
        # negative support vector hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min,hyp_x_max],[nsv1,nsv2], 'k')

        # (w.x+b) = 0
        # positive support vector hyperplane
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min,hyp_x_max],[db1,db2], 'y--')

        plt.show()
        
data_dict = {-1:np.array([[1,7],
                          [2,8],
                          [3,8],]),
             
             1:np.array([[5,1],
                         [6,-1],
                         [7,3],])}

svm = Support_Vector_Machine()
svm.fit(data=data_dict)

predict_us = [[0,10],
              [1,3],
              [3,4],
              [3,5],
              [5,5],
              [5,6],
              [6,-5],
              [5,8]]

for p in predict_us:
    svm.predict(p)

svm.visualize()


# ### Clustering

# In[1]:


from sklearn.cluster import KMeans


# In[11]:


X = np.array([[1,2],
              [4,5],
              [7,4],
              [2,2],
              [1,9],
              [6,6]]) 
clf = KMeans(n_clusters = 2)
clf.fit(X)
centroids = clf.cluster_centers_
labels = clf.labels_
colors = ["g.","r."]
for i in range (len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]],markersize = 25)
plt.scatter(centroids[:,0],centroids[:,1],marker ='x',s=150,linewidths=5)
plt.show()


# In[13]:


df = pd.read_csv("https://raw.githubusercontent.com/Tanmay451/Machine-Learning/master/titanic.csv")
df.drop(['body','name','home.dest'],1,inplace=True)
df.fillna(0,inplace=True)

df.head()


# In[4]:


df['sex'].count


# In[14]:


def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df

df = handle_non_numerical_data(df)
print(df.head())


# In[15]:


df = df.drop(["boat","sex"],1)
print(df.head())
X = np.array(df.drop(["survived"],1).astype(float))
X = preprocessing.scale(X)
y = np.array(df["survived"])


# In[26]:


clf = KMeans(n_clusters=2)
clf.fit(X)


# In[27]:


correct = 0
for i in range (len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1,len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct +=1
print(correct/len(X))


# In[ ]:





# In[ ]:




