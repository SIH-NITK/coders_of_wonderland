from skimage.measure import compare_ssim
import argparse
import cv2
import numpy as np
import scipy
import glob
import sys
import os
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

source_path  = sys.argv[1]

def stretch_contrast(img):
    norm_img1 = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_img2 = cv2.normalize(img, None, alpha=0, beta=1.2, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_img1 = (255*norm_img1).astype(np.uint8)
    norm_img2 = np.clip(norm_img2, 0, 1)
    norm_img2 = (255*norm_img2).astype(np.uint8)

    return norm_img2

def find_ssim_array(source_path):
    image_files = [source_path + '/' + f for f in glob.glob('*.tif')]
    file_names=[f for f in glob.glob('*.tif')]
    ssim_array=[]
    common_image="ground_image.tif"
    imageA = cv2.imread(common_image)
    imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    imageA = stretch_contrast(imageA)
    for image in image_files:
        image_name=image.split('/')[-1]
        if(common_image==image_name):
            continue
        imageB = cv2.imread(image)
        imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
        imageB = stretch_contrast(imageB)
        (score, diff) = compare_ssim(imageA, imageB, full=True)
        ssim_array.append(1-score)
    return ssim_array

def find_peaks(ssim_array):
    peaks=scipy.signal.find_peaks(ssim_array)
    prominance,left_bases,right_bases=scipy.signal.peak_prominences(ssim_array, peaks[0].tolist())
    new_peaks=[]
    for i in range(len(peaks[0])):
        if(prominance[i]>0.1):
            new_peaks.append(peaks[0][i]+1)
    return new_peaks


def pca(abs_path):
    pca= PCA(n_components=10)
    path=abs_path
    files=os.listdir(path)
    #print(files)
    data=[]
    for i in range(len(files)):
        #print(path+files[i])
        img=cv2.imread(path+files[i],0)
        #print(img.shape)
        data.append(img)
    var=[]
    single=[]
    for i in range(len(data)):
        pca.fit(data[i])
        var.append(pca.explained_variance_ratio_)
        single.append(pca.singular_values_)
    x=np.linspace(1,48,48)
    var=np.array(var)
    single=np.array(single)
    avg=np.mean(var,axis=1)
    peaks=scipy.signal.find_peaks(avg)[0]
    prominance,left_bases,right_bases=scipy.signal.peak_prominences(avg, peaks.tolist())
    ans=[]
    for i in range(len(prominance)):
        if(prominance[i]>0.025):
            ans.append((peaks[i]+1))
    
    return ans

def calc_darkness(a):
    x=0
    temp = 0
    for i in range(len(a)):
        temp = max(np.histogram(a[i],bins=3)[0][0],temp)
    return temp

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


def index_to_date(index,source_path):
    dict={}
    path=source_path
    files=os.listdir(path)
    list=["January","February","March","April","May","June","July","August","September","October","November","December"]
    date= files[index].split('_')[2]
    year=date[:4]
    month=date[4:]
    day=int(files[index].split('_')[4])
    if(day==1):
        val=1
    else:
        val=25
    str_1=str(val)+" "+list[int(month)-1]+" "+year
    return str_1

def main():
    os.chdir(source_path)

    ssim_array=find_ssim_array(source_path)
    output_ssim=find_peaks(ssim_array)
    output_pca=pca(source_path)

    kmeans_out=kmeans_periods(source_path)
    # print(kmeans_harvest)

    kmeans_sow=kmeans_out[1]

    for i in range(len(kmeans_sow)):
        print("Transistion occoured between",index_to_date(kmeans_sow[i][1],source_path),"and",index_to_date(kmeans_sow[i][2],source_path),"So the sowing has taken place between",index_to_date(kmeans_sow[i][0],source_path),"and",index_to_date(kmeans_sow[i][1],source_path))
    # print(kmeans_sow)

    kmeans_harvests=kmeans_out[0]
    kmeans_harvests = np.array(kmeans_harvests).T.tolist()

    kmeans_begin=kmeans_harvests[0]
    kmeans_end=kmeans_harvests[1]
    
    # print(kmeans_begin)
    # print(kmeans_end)
    # print(output_pca)
    # print(output_ssim)

    number_of_harvests=len(output_pca)
    print("Number of Harvests : ",number_of_harvests)
    #arr=np.append(kmeans_harvest,pca_output,ssim_output)
    for i in range(number_of_harvests):
        begin_harvest=min((output_pca[i],output_ssim[i],kmeans_begin[i]))
        end_harvest=max((output_pca[i],output_ssim[i],kmeans_end[i]))
        print("Harvest No.",str(i),"begins at : \t",index_to_date(begin_harvest,source_path))
        print("Harvest No.",str(i),"ends at : \t",index_to_date(end_harvest,source_path))
    # return output_harvests

print(main())