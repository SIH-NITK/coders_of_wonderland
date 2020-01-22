import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

def calc_darkness(a):
    x=0
    temp = 0
    for i in range(len(a)):
        temp = max(np.histogram(a[i],bins=3)[0][0],temp)
    return temp
    #return x/len(a)    
    
def kmeans_periods(abs_path):
    path=abs_path
    hist1=[]
    data=[]
    f=os.listdir(path)
    for i in range(len(f)):
        img=cv2.imread(path+f[i],0)
        data.append(img)
        a,b=np.histogram(img,bins=10)
        hist1.append(a)
    k=KMeans(n_clusters=3)
    k.fit(hist1)
    vector=k.predict(hist1)
    t1=[]#0
    t2=[]#1
    t3=[]#2
    for i in range(len(vector)):
        if vector[i]==0:
            t1.append(data[i])
        elif vector[i]==1:
            t2.append(data[i])
        else:
            t3.append(data[i])
            
    d1=calc_darkness(t1)
    d2=calc_darkness(t2)
    d3=calc_darkness(t3)
    d=[d1,d2,d3]
    final_vector=[]
    for i in range(len(vector)):
        if vector[i]==np.argmax(d):
            final_vector.append(2)
        elif vector[i]==np.argmin(d):
            final_vector.append(0)
        else:
            final_vector.append(1)
            
    sowing = []
    harvesting = []
    for i in range(0,len(final_vector)-1):
        if(final_vector[i]==2 and final_vector[i+1]==1):
            temp = []
            temp.append(i+1)
            temp.append(i+2)
            harvesting.append(temp)
    for i in range(0,len(final_vector)-2):
        if(final_vector[i]==0 and final_vector[i+1]==0 and final_vector[i+2]==1):
            temp = []
            temp.append(i+1)
            temp.append(i+2)
            temp.append(i+3)
            sowing.append(temp) 
    return harvesting,sowing
            
            