#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Code to split xyz to three diffrent groups
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import pyplot
from numpy import linalg as LA


# In[2]:


#Load data from mesh
#Total will be (x,y,z) from mesh
#First split total data to train and test
#Second import new train dataset and split it to train and validation
pd_noisefree=pd.read_csv('Total.csv')
position=pd_noisefree.values


# In[3]:


#Split data to train and test 
#split xyz
X_train, X_test=train_test_split(position[:,:3],test_size=0.2)


# In[1]:


#Write train xyz in csv format
Trainxyz=pd.DataFrame(X_train)                         
Trainxyz_copy = Trainxyz.copy()
Trainxyz_copy.to_csv('Train.csv')
#Write test xyz in csv format
Testxyz=pd.DataFrame(X_test)     
Testxyz_copy = Testxyz.copy()
Testxyz_copy.to_csv('Tset.csv') 




























# In[ ]:





# In[ ]:





# In[ ]:




