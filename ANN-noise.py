#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import pyplot
from tensorflow.keras.layers.experimental import preprocessing
normalizer = preprocessing.Normalization()


# In[5]:


Count1= keras.Input(shape=(1,), name="Count1")
Count2= keras.Input(shape=(1,), name="Count2")
Count3= keras.Input(shape=(1,), name="Count3")
Count4= keras.Input(shape=(1,), name="Count4")
Count5= keras.Input(shape=(1,), name="Count5")
Count6= keras.Input(shape=(1,), name="Count6")
Count7= keras.Input(shape=(1,), name="Count7")
Count8= keras.Input(shape=(1,), name="Count8")
Count9= keras.Input(shape=(1,), name="Count9")
Count10= keras.Input(shape=(1,), name="Count10")
Count11= keras.Input(shape=(1,), name="Count11")
Count12= keras.Input(shape=(1,), name="Count12")
Count13= keras.Input(shape=(1,), name="Count13")
Count14= keras.Input(shape=(1,), name="Count14")
Count15= keras.Input(shape=(1,), name="Count15")
Count16= keras.Input(shape=(1,), name="Count16")
Count17= keras.Input(shape=(1,), name="Count17")
Count18= keras.Input(shape=(1,), name="Count18")
Count19= keras.Input(shape=(1,), name="Count19")
Count20= keras.Input(shape=(1,), name="Count20")


# In[6]:


x = layers.concatenate([Count1,Count2,Count3,Count4,Count5,Count6,Count7,Count8,Count9,Count10,Count11,Count12,Count13,Count14,Count15,Count16,Count17,Count18,Count19,Count20])
hidden1 = layers.Dense(256, activation='relu')(x)
hidden2 = layers.Dense(512, activation='relu')(hidden1)
hidden3 = layers.Dense(1024, activation='relu')(hidden2)


# In[7]:


X = layers.Dense(1,activation='linear', name="xx")(hidden3)
Y = layers.Dense(1,activation='linear', name="yy")(hidden3)
Z = layers.Dense(1,activation='linear', name="zz")(hidden3)


# In[8]:


model = keras.Model(inputs=[Count1,Count2,Count3,Count4,Count5,Count6,Count7,Count8,Count9,Count10,Count11,Count12,Count13,Count14,Count15,Count16,Count17,Count18,Count19,Count20],outputs=[X, Y,Z],)


# In[9]:


keras.backend.set_epsilon(1)
model.compile(
    optimizer='adam',
    loss=['mse', 'mse','mse'],
    loss_weights=[1.0, 1.0,1.0],
    metrics=['MAE']
     
)


# In[10]:


#To randomly splitting Train/test/ validation use the Splittting code
#Import Train data
pd_dat=pd.read_csv('TrainNoisyFeed.csv')
TrainNoisy=pd_dat.values
#Import Test data
pd_test=pd.read_csv('TestNoisyFeed.csv')
TestNoiseFree=pd_test.values


# In[11]:


X_train=TrainNoisy[:,:20]
Y_train=TrainNoisy[:,20:]
X_test=TestNoiseFree[:,:20]
Y_test=TestNoiseFree[:,20:]


# In[9]:


count1_train, count2_train, count3_train, count4_train,count5_train,count6_train,count7_train,count8_train,count9_train,count10_train,count11_train,count12_train,count13_train,count14_train,count15_train,count16_train,count17_train,count18_train,count19_train,count20_train=np.transpose(X_train)
count1_test, count2_test, count3_test, count4_test,count5_test,count6_test,count7_test,count8_test,count9_test,count10_test,count11_test,count12_test,count13_test,count14_test,count15_test,count16_test,count17_test,count18_test,count19_test,count20_test=np.transpose(X_test)


# In[10]:


x_train, y_train, z_train= Y_train[:,0], Y_train[:,1], Y_train[:,2]
x_test, y_test, z_test= Y_test[:,0], Y_test[:,1], Y_test[:,2]


# In[11]:


inputs_train=[count1_train, count2_train, count3_train, count4_train,count5_train,count6_train,count7_train,count8_train,count9_train,count10_train,count11_train,count12_train,count13_train,count14_train,count15_train,count16_train,count17_train,count18_train,count19_train,count20_train]
outputs_train=[x_train,y_train,z_train]


# In[12]:


#Load Validation data set
pd_val=pd.read_csv('ValidationFeed.csv')
Validation=pd_val.values
X_val=Validation[:,:20]
Y_val=Validation[:,20:]
count1_val, count2_val, count3_val, count4_val,count5_val,count6_val,count7_val,count8_val,count9_val,count10_val,count11_val,count12_val,count13_val,count14_val,count15_val,count16_val,count17_val,count18_val,count19_val,count20_val=np.transpose(X_val)
x_val, y_val, z_val= Y_val[:,0], Y_val[:,1], Y_val[:,2]
inputs_val=[count1_val, count2_val, count3_val, count4_val,count5_val,count6_val,count7_val,count8_val,count9_val,count10_val,count11_val,count12_val,count13_val,count14_val,count15_val,count16_val,count17_val,count18_val,count19_val,count20_val]
outputs_val=[x_val,y_val,z_val]


# In[13]:


history=model.fit(inputs_train,outputs_train,
                 epochs=1000,
                 batch_size=64,
                validation_data=(inputs_val, outputs_val)
                 
                 )


# In[14]:


result=model.evaluate([count1_test,count2_test,count3_test,count4_test,count5_test,count6_test,count7_test,count8_test,count9_test,count10_test,count11_test,count12_test,count13_test,count14_test,count15_test,count16_test,count17_test,count18_test,count19_test,count20_test],[x_test,y_test,z_test],verbose=2)
print(result)


# In[15]:


pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='validation')
pyplot.legend()


pd_pre=pd.read_csv('Prediction.csv')
datasetpre=pd_pre.values
count1_pre, count2_pre, count3_pre, count4_pre,count5_pre,count6_pre,count7_pre,count8_pre,count9_pre,count10_pre,count11_pre,count12_pre,count13_pre,count14_pre,count15_pre,count16_pre,count17_pre,count18_pre,count19_pre,count20_pre=np.transpose(datasetpre)
prediction=model.predict([count1_pre, count2_pre, count3_pre, count4_pre,count5_pre,count6_pre,count7_pre,count8_pre,count9_pre,count10_pre,count11_pre,count12_pre,count13_pre,count14_pre,count15_pre,count16_pre,count17_pre,count18_pre,count19_pre,count20_pre])


Result=pd.DataFrame([prediction])     
Result_copy = Result.T.copy()
Result_copy.to_csv('predictionresult.csv') 


# In[16]:


pyplot.subplot(211)
pyplot.title('Loss')
plt.semilogy(history.history['loss'], label='train')
plt.semilogy(history.history['val_loss'], label='validation')
pyplot.legend()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




