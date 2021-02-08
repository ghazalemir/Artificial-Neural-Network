# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 12:34:06 2021

@author: ghaza
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA
import pandas as pd



def depth_calculation(detector_position):
    
    depth_vector=[]
    distance=[]
    pd_noisefree=pd.read_csv('FileName.csv')
    final=pd_noisefree.values
    
    



    for i in range(0,len(final)):
        distance_vector=[]
        for j in range(0,3):
            distance_vector.append(final[i][j]-detector_position[j])
        depth_vector.append(distance_vector)


    
    for i in range(0,len(final)): 
        distance.append(np.fabs(LA.norm(depth_vector[i])))
        
    
    return distance
"""
"""
def Number_count(detector_position):
    distance=depth_calculation(detector_position)
    
    count=[]
    for i in range (0,len(distance)):
        count.append(1/distance[i])
    #print(count)
    return count

      




Count1=Number_count([1.2,0,0.1])
Count2=Number_count([-1.2,0,0.1])
Count3=Number_count([0,1.2,0.1])
Count4=Number_count([0,-1.2,0.1])
Count5=Number_count([1.2,0,0.3])
Count6=Number_count([-1.2,0,0.3])
Count7=Number_count([0,1.2,0.3])
Count8=Number_count([0,-1.2,0.3])
Count9=Number_count([1.2,0,0.5])
Count10=Number_count([-1.2,0,0.5])
Count11=Number_count([0,1.2,0.5])
Count12=Number_count([0,-1.2,0.5])
Count13=Number_count([1.2,0,0.7])
Count14=Number_count([-1.2,0,0.7])
Count15=Number_count([0,1.2,0.7])
Count16=Number_count([0,-1.2,0.7])
Count17=Number_count([1.2,0,0.9])
Count18=Number_count([-1.2,0,0.9])
Count19=Number_count([0,1.2,0.9])
Count20=Number_count([0,-1.2,0.9])




Result=pd.DataFrame([Count1,Count2,Count3,Count4,Count5,Count6,Count7,Count8,Count9,Count10,Count11,Count12,Count13,Count14,Count15,Count16,Count17,Count18,Count19,Count20])     
Result_copy = Result.T.copy()
Result_copy.to_csv('NewFileName.csv')    


    
