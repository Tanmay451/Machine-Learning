#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#print(x_train[0])
#plt.imshow(x_train[0])
#print(y_train[0])
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)
#print(x_train[0])
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 3)
val_loss, val_acc = model.evaluate(x_test, y_test)


# In[2]:


model.save('mnist_model.model')


# In[3]:


new_model = tf.keras.models.load_model('mnist_model.model')


# In[9]:


predictions = new_model.predict(x_test[:9999])


# In[10]:


plt.imshow(x_test[9998])
print(np.argmax(predictions[9998]))


# In[ ]:




