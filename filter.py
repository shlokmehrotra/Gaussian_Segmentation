#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 22:48:22 2019

@author: shlokmehrotra
"""


import numpy as np
import cv2 as cv2
#import libtiff
#import PIL.Image as Image


#img = cv2.imread("dataset.jpg", 1)

img = cv2.imread('dataset-1.tif', 1)
print(img)
gray = cv2.imwrite('out.tif', img)
#gray = cv2.imread('out.tif', -1)
#print(gray)

print(img.dtype)
print(img.shape)

blue = cv2.imwrite("blue.tif", img[:,:,0])
green = cv2.imwrite("green.tif", img[:,:,1])
red = cv2.imwrite("red.tif", img[:,:,2])

spot = cv2.imread("green.tif")

spot_detect = cv2.imwrite("blurspot.tif", spot)

#thresh = cv2.threshold(spot_detect, 200, 255, cv2.THRESH_BINARY)[1]


#RIHJT HERE

im = "blurspot.tif"
# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector()
 
# Setup SimpleBlobDetector parameters.
#params = cv2.SimpleBlobDetector_Params()
 
# Change thresholds
params.minThreshold = 30;
params.maxThreshold = 245;
 
# Filter by Area.
params.filterByArea = False
#params.minArea = 1500
 
# Filter by Circularity
params.filterByCircularity = False
#params.minCircularity = 0.1
 
# Filter by Convexity
params.filterByConvexity = False
#params.minConvexity = 0.87
 
# Filter by Inertia
params.filterByInertia = False
#params.minInertiaRatio = 0.01
 
# Create a detector with the parameters


detector = cv2.SimpleBlobDetector_create(params)



# Detect blobs.
keypoints = detector.detect(np.array(im))
 
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob


im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
# Show keypoints


cv2.imwrite("final.tif", im_with_keypoints)



#kernel = np.ones((5,5),np.uint8)
#denoise = cv2.erode(green,kernel,iterations = 1)


#processed = cv2.imwrite("processed.jpg", denoise)

#print(img)
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


#cv2.imwrite("orig.jpg", img)
#blur = cv2.blur(gray, (2,2))
#cv2.imwrite("blur.jpg", blur)

#minVal, maxVal, minLoc, maxLoc = (cv2.minMaxLoc(blur))

#print(minLoc, maxLoc)
#cv2.circle(img, maxLoc, 2, (0,0,255), 2)

#cv2.imwrite("processed.jpg", img)

