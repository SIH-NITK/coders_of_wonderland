# import the necessary packages
from skimage.measure import compare_ssim
import argparse
import cv2
import numpy as np

def stretch_contrast(img):
# img = cv2.imread("zelda3_bm20_cm20.jpg", cv2.IMREAD_COLOR)

# normalize float versions
    norm_img1 = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_img2 = cv2.normalize(img, None, alpha=0, beta=1.2, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # scale to uint8
    norm_img1 = (255*norm_img1).astype(np.uint8)
    norm_img2 = np.clip(norm_img2, 0, 1)
    norm_img2 = (255*norm_img2).astype(np.uint8)

    return norm_img2

# construct the argument parse and parse the arguments
import glob

source_path = "C:\\Users\\Mukesh_2\\Downloads\\Clipped_NDVI"

image_files = [source_path + '/' + f for f in glob.glob('*.tif')]
file_names=[f for f in glob.glob('*.tif')]

ssim_array=[]

common_image="awifs_ndvi_201801_15_2_clipped.tif"
file_names.remove(common_image)
imageA = cv2.imread("awifs_ndvi_201801_15_2_clipped.tif")
#imageA=imageA.reshape((imageA.shape[1],imageA.shape[0],imageA.shape[2]))
#print(imageA.shape)
imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
# #grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# imageA = clahe.apply(imageA)
imageA = stretch_contrast(imageA)

for image in image_files:
    print(image)
    image_name=image.split('/')[-1]
    
    if(common_image==image_name):
        continue
    
    imageB = cv2.imread(image)
    #imageB=imageB.reshape((imageB.shape[1],imageB.shape[0],imageB.shape[2]))
    imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imageB = stretch_contrast(imageB)
    # convert the images to grayscale
    # grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    # grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    (score, diff) = compare_ssim(imageA, imageB, full=True)
    ssim_array.append(score)

for i in range(len(image_files)):
        print(ssim_array[i],image_files[i])

import numpy as np
import matplotlib.pyplot as plt

# x = np.arange(0, 5, 0.1)
# y = np.sin(x)
plt.plot(file_names,ssim_array)
plt.show()