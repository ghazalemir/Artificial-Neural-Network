# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 17:12:06 2021

@author: ghaza
"""

import pandas as pd
import numpy as np 
from numpy import linalg as LA


pd_noisefree=pd.read_csv('Validation.csv')
NoiseFree=pd_noisefree.values



mu, sigma = 0, 0.00265

# creating a noise with the same dimension as the dataset (2,2) 
noise = (np.random.normal(mu, sigma, [12800,3]))

"""
distance=[]
for i in range(0,len(noise)):
    distance.append(LA.norm(noise[i])*1000)
    

print(distance)
    
dis=0
for i in range (0,len(distance)):
    dis=dis+distance[i]
    
ave=dis/len(distance)

print("ave")
print(ave)
"""    

NoiseFree = NoiseFree + noise

Result=pd.DataFrame(NoiseFree)     
Result_copy = Result.copy()
Result_copy.to_csv('TestNoisyFeed.csv') 
            


