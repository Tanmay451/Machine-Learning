{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import quandl\n",
    "import math\n",
    "import numpy as np\n",
    "from sklearn import preprocessing, svm\n",
    "from sklearn.model_selection import cross_validate\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LinearRegression\n",
    "import datetime\n",
    "import matplotlib.pyplot as plt\n",
    "from matplotlib import style"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = quandl.get(\"WIKI/GOOGL\")\n",
    "print(df.head(3))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]\n",
    "print(df.head(3))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['HL. PCT'] = (df['Adj. High'] - df['Adj. Close'])/df['Adj. Close']* 100.0\n",
    "df['PIC_change'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open']* 100.0\n",
    "df = df[['Adj. Close','HL. PCT','PIC_change','Adj. Volume']] "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "print(df.head(3))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "forecast_col = 'Adj. Close'\n",
    "df.fillna(-99999, inplace = True)\n",
    "forecast_out = int(math.ceil(0.01*len(df)))\n",
    "df['label'] = df[forecast_col].shift(-forecast_out)\n",
    "df.dropna(inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(df.head(3))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = np.array(df.drop(['label'],1))\n",
    "y = np.array(df['label'])\n",
    "X = preprocessing.scale(X)\n",
    "y = np.array(df['label'])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)\n",
    "\n",
    "clf = LinearRegression()\n",
    "# to make training faster just LinearRegression(n_jods=10)\n",
    "clf.fit(X_train, y_train)\n",
    "accuracy = clf.score(X_test,y_test)\n",
    "print(accuracy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#another algo for classification but it is not so accurate \"support vector regration'\n",
    "clf = svm.SVR()\n",
    "clf.fit(X_train, y_train)\n",
    "accuracy = clf.score(X_test,y_test)\n",
    "print(accuracy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# if we want to save our classifier to avoid procassing time every time we use it\n",
    "# we should do it after training with data\n",
    "import pickle\n",
    "clf = LinearRegression(n_jobs=-1)\n",
    "clf.fit(X_train, y_train)\n",
    "with open('linearRegration.pickle','wb') as f:\n",
    "    pickle.dump(clf,f)\n",
    "#till now we have load our data and now we will reading it......... Once we train we can comment out above code or the training part and just use the file instade\n",
    "pickle_in = open('linearRegration.pickle','rb')\n",
    "clf = pickle.load(pickle_in)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_lately = X[-forecast_out:]\n",
    "X = X[:-forecast_out]\n",
    "\n",
    "df.dropna(inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(last_date.head(3))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "forecast_set = clf.predict(X_lately)\n",
    "df['Forecast'] = np.nan\n",
    "\n",
    "last_date = df.iloc[:0:-1].name\n",
    "last_unix = last_date.timestamp()\n",
    "one_day = 86400\n",
    "next_unix = last_unix + one_day\n",
    "\n",
    "for i in forecast_set:\n",
    "    next_date = datetime.datetime.fromtimestamp(next_unix)\n",
    "    next_unix += 86400\n",
    "    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]\n",
    "\n",
    "df['Adj. Close'].plot()\n",
    "df['Forecast'].plot()\n",
    "plt.legend(loc=4)\n",
    "plt.xlabel('Date')\n",
    "plt.ylabel('Price')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
