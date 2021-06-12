#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:38:36 2019

@author: shlokmehrotra
"""
    
from PIL import Image
import numpy as np
import csv
import cv2
from numpy import linspace
from pylab import plot,show,hist,figure,title
from scipy.stats import norm
from scipy import optimize
import scipy.optimize as opt
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from mpl_toolkits import mplot3d


def read_tiff1(path, time_intervals, stack_size):
    """
    path - Path to the multipage-tiff file
    n_images - Number of pages in the tiff file
    """
    img = Image.open(path)
    #meta data of tiff modified such that only grayscale image values appear -> previously in the thousands
    print(img.height, img.width)
    num_stacks = time_intervals
    #num_stacks = int(n_images/stack_size) -> given total amount of frames
    images = []
    for i in range(num_stacks):
        val = []
        for j in range(stack_size):
            
            try:
                img.seek(i)
                pixval = list(img.getdata())
                val.append(pixval)
                
            except EOFError:
                # Not enough frames in img
                break
        print(i ," is done")
        images.append(val)
    return(np.array(images))
#the below code calculates brightest points in each z stack and flattens into 2D in which spots are identified
def identifySpotLoc(data, time_intervals, stack_size, frame_area):
    brightest = []
    for i in range(time_intervals):
        val = []
        for j in range(frame_area):
            pix_bright = []
            for k in range(stack_size):
                pix_bright.append((data[i][k][j]))
            val.append(max(pix_bright))
        brightest.append(val)
    #brightest contains the 2D elements of all the brightest spots in a stack
    return(brightest)
    
    
def findSpots(brightarr, xlen, ylen):
    #Handles a 2D array of brightest pixels in a stack and finds the (x,y) coordinates of suspected spots using openCV
    brightarr = np.array(brightarr)
    brightarr.reshape((xlen, ylen))
    print(brightarr.shape)
#time -> z-stack -> pixel
#but i want to iterate through each stack  for pixel val
#Calling function to reaad tiff file
#read_tiff("imagepath", totalFrames, Zstack)
data = read_tiff1("Series011.tif", 29, 19)

with open('imageData.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(data)
writeFile.close()
print(data.size)
print(data[0].size) #At time = 0;
print(data[0][0].size) #At time = 0 and z stack of 0

spotFinds = identifySpotLoc(data, 29, 19, 589824)
print("Size of spotFinds: ", len(spotFinds)) #should be 768 * 768 * 29
print("spotarray: ", len(spotFinds[0]))

with open('trash.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(spotFinds)
writeFile.close()
print("Encontré todos los puntos vivos")
#[[[im]....(zstack)]....[[im]....(zstack)].....(time)]
