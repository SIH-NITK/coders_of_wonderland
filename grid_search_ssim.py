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

i=0
for image in image_files:
    common_image=image_name=image.split('/')[-1]
    # file_names.remove(common_image)
    imageA = cv2.imread(common_image)
    imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    imageA = stretch_contrast(imageA)

    for image_1 in image_files:
        # print(image)
        image_name_1=image_1.split('/')[-1]
        # print(image_name,common_image)
        if(common_image==image_name_1):
            continue
        
        imageB = cv2.imread(image_1)
        imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
        imageB = stretch_contrast(imageB)
        (score, diff) = compare_ssim(imageA, imageB, full=True)
        print(i,(1-score,common_image,image_name_1))
        ssim_array.append((1-score,common_image,image_name_1))
        i+=1
from operator import itemgetter
first_item = itemgetter(0)
ssim_array = sorted(ssim_array, key = first_item)

print("-------------------------------------------------------------------------------------------")
for i in ssim_array:
    print(i)
# for i in range(len(image_files)-1):
#         print(ssim_array[i],image_files[i])

# import numpy as np
# import matplotlib.pyplot as plt

# # x = np.arange(0, 5, 0.1)
# # y = np.sin(x)
# plt.plot(file_names,ssim_array)
# plt.show()