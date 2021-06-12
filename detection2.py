#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#"""
#Created on Mon Jul  1 12:09:25 2019

#@author: shlokmehrotra
#"""
#import pymc3 as pm
import cv2
import csv
import numpy as np
from numpy import linspace
import matplotlib.pyplot as plt
from pylab import plot,show,hist,figure,title
from scipy.stats import norm
from scipy import optimize
import scipy.optimize as opt
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from mpl_toolkits import mplot3d
#from gaus import *

# Read image
im = cv2.imread("blurspot.tif", cv2.IMREAD_GRAYSCALE)

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 40
params.maxThreshold = 200

#looking for spots that tend towards lighter colors
params.blobColor = 255

# Filter by Area.
params.filterByArea = True
params.minArea = 10
params.maxArea = 2500

# Filter by Circularity (Future Reference if needed)
params.filterByCircularity = False
#params.minCircularity = 0.1

# Filter by Convexity (Future Reference if needed)
params.filterByConvexity = False
#params.minConvexity = 0.87
    
# Filter by Inertia (Future Reference if needed)
params.filterByInertia = False
#params.minInertiaRatio = 0.01

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)


# Detect blobs.
keypoints = detector.detect(im)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob

im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show blobs
cv2.imwrite("final.tif", im_with_keypoints)


final = cv2.imread("final.tif")


#cv2.waitKey(0)
xlist = []
ylist = []
sizelist = []

data = []
reader = cv2.imread("blurspot.tif")
height = reader.shape[0]
width = reader.shape[1]

#print(height, width)

#dimensions of box size are 2 * boxsize by 2 * boxsize 
boxsize = 15

for keypoint in keypoints:
    #print("x is ", keypoint.pt[0], " y is ", keypoint.pt[1], " and the size is ", keypoint.size)
    xlist.append(round(keypoint.pt[0]))
    ylist.append(round(keypoint.pt[1]))
    sizelist.append(keypoint.size)
    if(xlist[-1] + boxsize > width or xlist[-1] - boxsize < 0 or ylist[-1] + boxsize > height or ylist[-1] - boxsize < 0):
        del xlist[-1]
        del ylist[-1]


for number in range(len(xlist)):
    filename = "data/image#" + str(number) + ".tif"


    extracted_data = reader[ylist[number] - boxsize: ylist[number] + boxsize, xlist[number] - boxsize: xlist[number] + boxsize]
    #for i in range(len(extracted_data)):
    #    for j in range(len(extracted_data[0])):
    #        somearr = extracted_data[i][j]
    #        np.delete(extracted_data[i][j], [1,2])
    #print(extracted_data)
    #print(extracted_data[0][0])
    data.append(extracted_data)
    final[ylist[number] - boxsize: ylist[number] + boxsize, xlist[number] - boxsize: xlist[number] + boxsize] = (0,0,255)
#final[7:140,100:400] = (0,0,255)
    cv2.imwrite(filename, extracted_data)
    #info = cv2.imread(filename)
    #print(info.shape)
cv2.imwrite("redblobs.tif", final)

newdata = []
for info in data:
    subdata = []
    #print(len(info))
    for bar in info:
        subsubdata = []
        for jr in bar:
            bit = (jr[0])/255 #- 28/255            #offset value ( - 28 ) - subject to change depending on contrast
            if(bit < 0): 
                bit = 0
            subsubdata.append(bit) 
            
            #print(jr[0])
        subdata.append(subsubdata)
    newdata.append(subdata)
maxn = np.max(newdata)
minn = np.min(newdata)
#print(maxn, minn)
#newdata contains all the data for all the images with brightness levels in a (boxsize by boxsize) 2D array
#print(newdata)

newdata = np.array(newdata)
#data of first image(2D matrix)
testdata = newdata[0]
fig = plt.figure(figsize=(8, 4))


plt.imshow(testdata)

plt.colorbar(orientation='vertical')
plt.show()



with open('data.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(testdata)
writeFile.close()


def simulateOneGaus(datapoints):
    
    distarr = []
    gausarr = []
    heightreal1 = []
    spotloc1 = []
    def gaus2d(x, y, mx, my, sx, sy, height):
        return(mx, my, height * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.))))
    
    x = np.linspace(0, 29, 30)  
    y = np.linspace(0, 29, 30)
    x, y = np.meshgrid(x, y) # get 2D variables instead of 1D
    
    #np.random.seed(123)
    
    #number of datapoints
    for i in range(datapoints):
        
        random = np.random.random(14)
        mx = 17 + random[0] * 7
        my = 15
        
        
        height = 0.2 + random[4] * 0.3
        
        sx = 2 + random[6] * 1
        sy = 2 + random[7] * 1
        
        
        x1,y1, z = gaus2d(x, y, mx, my, sx, sy, height)
        #dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        #print(x1, x2, y1, y2)
        spotloc1.append([x1,y1])
        #simulated data noise factor
        #distarr.append(dist)
        for i in range(30):
            for j in range(30):
                z[i][j] *= np.random.uniform(0.95, 1.05, 1)

        gausarr.append(z)
        
        #one spot
        #heightsum.append(height)
        #two spot
        heightreal1.append(height)
        
        plt.figure(figsize=(8, 4))


        plt.imshow(z)

        plt.colorbar(orientation='vertical')
        plt.show()
    #print("2D Gaussian-like array:")
    #print(z)
    plt.figure(figsize=(8, 4))


    plt.imshow(z)

    plt.colorbar(orientation='vertical')
    plt.show()
    return(spotloc1, gausarr, heightreal1)






def simulateGaus(datapoints):
    distarr = []
    gausarr = []
    heightreal1 = []
    heightreal2 = []
    spotloc1 = []
    spotloc2 = []
    def gaus2d(x, y, mx, my, sx, sy, height,  mx1, my1, sx1, sy1, height1):
        return(mx,mx1,my,my1, height * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.))) + height1 * np.exp(-((x - mx1)**2. / (2. * sx1**2.) + (y - my1)**2. / (2. * sy1**2.))))
    
    x = np.linspace(0, 29, 30)  
    y = np.linspace(0, 29, 30)
    x, y = np.meshgrid(x, y) # get 2D variables instead of 1D
    
    #np.random.seed(123)
    
    #number of datapoints
    for i in range(datapoints):
        
        random = np.random.random(14)
        mx = 17 + random[0] * 7
        my = 15
        
        
        height = 0.2 + random[4] * 0.3
        height1 = 0.2 + random[5] * 0.3
        
        sx = 2 + random[6] * 1
        sy = 2 + random[7] * 1
        
        sx1 = 2 + random[8] * 1
        sy1 = 2 + random[9] * 1    
        
        mx1 = 13 - random[10] * 7 
        my1 = 15 
        
        x1,x2,y1,y2, z = gaus2d(x, y, mx, my, sx, sy, height,  mx1, my1, sx1, sy1, height1)
        dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        #print(x1, y1, x2, y2)
        spotloc1.append([x1,y1])
        spotloc2.append([x2,y2])
        #simulated data noise factor
        distarr.append(dist)
        for i in range(30):
            for j in range(30):
                z[i][j] *= np.random.uniform(0.95, 1.05, 1)

        gausarr.append(z)
        
        #one spot
        #heightsum.append(height)
        #two spot
        heightreal1.append(height)
        heightreal2.append(height1)
        
        fig = plt.figure(figsize=(8, 4))


        plt.imshow(z)

        plt.colorbar(orientation='vertical')
        plt.show()
    #print("2D Gaussian-like array:")
    #print(z)
    fig = plt.figure(figsize=(8, 4))


    plt.imshow(z)

    plt.colorbar(orientation='vertical')
    plt.show()
    return(spotloc1, spotloc2, gausarr, distarr, heightreal1, heightreal2)
def TwoGausData(numero, theData):
    def gaussian(height, center_x, center_y, width_x, width_y, rotation, height1, center_x1, center_y1, width_x1, width_y1, rotation1):
            
        """Returns a gaussian function with the given parameters"""
        width_x = float(width_x)
        width_y = float(width_y)

        rotation = np.deg2rad(rotation)
        center_x = center_x * np.cos(rotation) - center_y * np.sin(rotation)
        center_y = center_x * np.sin(rotation) + center_y * np.cos(rotation)

        def rotgauss(x,y):
            xp = x * np.cos(rotation) - y * np.sin(rotation)
            yp = x * np.sin(rotation) + y * np.cos(rotation)
            g = height*np.exp(
                -(((center_x-xp)/width_x)**2+
                  ((center_y-yp)/width_y)**2)/2.)
            return g
        return rotgauss
        
        
# Standard Gaussian Fit w/o rotation==================================================
# 
#           """Returns a gaussian function with the given parameters"""
#           width_x = float(width_x)
#           width_y = float(width_y)
#           return lambda x,y: height*np.exp(
#                       -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)
# 
# =============================================================================
    #this should work
    def gaussian_norot(off1, off2, height, center_x, center_y, width_x, width_y, height1, center_x1, center_y1, width_x1, width_y1):
        """Returns a gaussian function with the given parameters"""
        width_x = float(width_x)
        width_y = float(width_y)
        width_x1 = float(width_x1)
        width_y1 = float(width_y1)
        return lambda x,y: off1 + height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2) + off2 + height1*np.exp(
                -(((center_x1-x)/width_x1)**2+((center_y1-y)/width_y1)**2)/2)
    #problematic
    def moments(data):
        """Returns  (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution by calculating its
        moments """
# =============================================================================
#         total = data.sum()
#         X, Y = np.indices(data.shape)
#         x = (X*data).sum()/total
#         y = (Y*data).sum()/total
#         col = data[:, int(y)]
#     
#     
#         width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
#         height = data.max()
#         height1 = height
# =============================================================================
        return(0, 0, 1, 15, 14, 3, 3, 1, 14, 16, 3, 2)
    
    
    #fitting routine
    def fitgaussian(data):
        """Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution found by a fit"""
        params = moments(data)
        errorfunction = lambda p: np.ravel(gaussian_norot(*p)(*np.indices(data.shape)) -
                                     data)
        #scipy rihjt
        # Levenberg-Marquadt algorithm -> leastsq
        #bnds = None
        off1, off2, height, x, y, width_x, width_y, he1, x1,y1, wi1, wi2  = params
        #p, success = optimize.leastsq(errorfunction, params)
        bnds = (0,30)
        p = optimize.least_squares(errorfunction, params, bounds = bnds).x
        
        #least square fitting(minimizes raw data and fit)

        if(p[2] < 1 and p[7] < 1 and p[3] > 0 and p[3] < 30 and p[4] > 0 and p[4] < 30 and p[8] > 0 and p[8] < 30 and p[9] > 0 and p[9] < 30):
             #print("pass case")
             return(p)
        else:
            print("failed case")
            print("height1", p[2],"height2", p[7], "X",  p[3],"Y",  p[4],"Y1",  p[8], "Y2", p[9])
            print("bounding error" + str(numero))        

        return p
    # Create the gaussian data
    Xin, Yin = np.mgrid[0:30, 0:30]
    data = gaussian_norot(*fitgaussian(theData))(Yin, Xin)
    #data_mean = (np.mean(theData.reshape(900,1)))
    #"data" vs "theData" data is processed while theData is still raw
    diffarray = []
    placehold = []
    
    for i in range(30):
        for j in range(30):
            placehold.append(data[i,j]- theData[i,j])
        diffarray.append(placehold)
        placehold = []      
        
    diffarray = np.array(diffarray)
    
    diffarray = diffarray.reshape(30,30)
    
    params = fitgaussian(data)
    fit = gaussian_norot(*params)
    (off1, off2, height, xspot, yspot, width_x, width_y, he1, x1spot,y1spot, width_x1, width_y1) = params
    distance = (np.sqrt((xspot - x1spot)**2 + (yspot - y1spot)**2))
# =============================================================================
#     plt.matshow(diffarray)
#     
#     
#     
#     
#     plt.contour(fit(*np.indices(data.shape)), cmap=plt.cm.copper)
#     ax = plt.gca()
#     (height, x, y, width_x, width_y, he1, x1,y1, wi1, wi2 ) = params
#     distance = (np.sqrt((x - x1)**2 + (y - y1)**2))
#     
#       
#     plt.text(0.95, 0.05, """
#     x : %.1f
#     y : %.1f
#     width_x : %.1f
#     width_y : %.1f""" %(x, y, width_x, width_y),
#             fontsize=12, horizontalalignment='right',
#             verticalalignment='bottom', transform=ax.transAxes)
#     
#     
# =============================================================================
    
    ax = plt.axes(projection='3d')
    
    
    #print(x)
    
    x = np.repeat(np.arange(30),30)
    y = np.tile(np.arange(30),30)
    #print(x)
    z = data
    
    X, Y, Z = (u.reshape(30, 30) for u in (x, y, z))
    
    ax.plot_surface(X, Y, Z, cmap="autumn_r", lw=0.5, rstride=1, cstride=1)
    ax.contour(X, Y, Z, 10,  cmap="autumn_r", linestyles="solid", offset=-1)
    ax.contour(X, Y, Z, 10, colors="k", linestyles="solid")

    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.title("Graph#" + str(numero))
    plt.gcf().text(0.02, 0.03, distance, fontsize=9)
    plt.savefig('3dgraphs2gaus/graph#' + str(numero) +  '.png')
    
    plt.close()
                
#plotting    
# =============================================================================
#     ax = plt.axes(projection='3d')
#     
#     difference = diffarray.copy()
#     difference = difference.reshape(900,1)
#     
#     
#     numdifference = np.std(difference)
#     #numdifference = np.mean(np.absolute(difference))
# 
# 
#     x = np.repeat(np.arange(30),30)
#     y = np.tile(np.arange(30),30)
# 
#     z = diffarray
#     
#     X, Y, Z = (u.reshape(30, 30) for u in (x, y, z))
#     
#     ax.plot_surface(X, Y, Z, cmap="autumn_r", lw=0.5, rstride=1, cstride=1)
#     ax.contour(X, Y, Z, 10,  cmap="autumn_r", linestyles="solid", offset=-1)
#     ax.contour(X, Y, Z, 10, colors="k", linestyles="solid")
#     
#     
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')
# =============================================================================
    #return(numdifference)
    off1, off2, height, xspot, yspot, width_x, width_y, he1, x1spot,y1spot, width_x1, width_y1
    return(off1, off2, width_x, width_y, width_x1, width_y1, xspot, yspot, x1spot, y1spot, height, he1, distance)


def OneGausData(numero, theData):
    def gaussian(off1, height, center_x, center_y, width_x, width_y, rotation):
            
        """Returns a gaussian function with the given parameters"""
        width_x = float(width_x)
        width_y = float(width_y)

        rotation = np.deg2rad(rotation)
        center_x = center_x * np.cos(rotation) - center_y * np.sin(rotation)
        center_y = center_x * np.sin(rotation) + center_y * np.cos(rotation)

        def rotgauss(x,y):
            xp = x * np.cos(rotation) - y * np.sin(rotation)
            yp = x * np.sin(rotation) + y * np.cos(rotation)
            g = off1 + height*np.exp(
                -(((center_x-xp)/width_x)**2+
                  ((center_y-yp)/width_y)**2)/2.)
            return g
        return rotgauss
    
        
        
# Standard Gaussian Fit w/o rotation==================================================
# 
#           """Returns a gaussian function with the given parameters"""
#           width_x = float(width_x)
#           width_y = float(width_y)
#           return lambda x,y: height*np.exp(
#                       -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)
# 
# =============================================================================
           
    def gaussian_norot(height, center_x, center_y, width_x, width_y):
        """Returns a gaussian function with the given parameters"""
        width_x = float(width_x)
        width_y = float(width_y)
        return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)
        
    def moments(data):
        """Returns  (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution by calculating its
        moments """
        total = data.sum()
        X, Y = np.indices(data.shape)
        x = int((X*data).sum()/total)
        y = int((Y*data).sum()/total)
        col = data[:, int(y)]
    
        width_x = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    
        row = data[int(x), :]
        width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
        height = data.max()
        return(0, height, x, y, width_x, width_y, 0.0)
        #return(1, 15, 15, 2, 2, 0.0)
    def fitgaussian(data):
        """Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution found by a fit"""
        params = moments(data)
        errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                     data)
        #p, success = optimize.leastsq(errorfunction, params)
        bnds = (0,30)
        p = optimize.least_squares(errorfunction, params, bounds = bnds).x
        #least square fitting(minimizes raw data and fit)
        return p
    # Create the gaussian data
    Xin, Yin = np.mgrid[0:30, 0:30]
    
    data = gaussian(*fitgaussian(theData))(Yin, Xin)
    data_mean = (np.mean(theData.reshape(900,1)))
    #"data" vs "theData" data is processed while theData is still raw
    diffarray = []
    placehold = []
    
    for i in range(30):
        for j in range(30):
            placehold.append(data[i,j]- theData[i,j])
        diffarray.append(placehold)
        placehold = []      
        
    diffarray = np.array(diffarray)
    
    diffarray = diffarray.reshape(30,30)
    
    params = fitgaussian(data)
    fit = gaussian(*params)
    
    
    off1, height, xspot, yspot, width_x, width_y, rot = params
# =============================================================================
#     plt.matshow(diffarray)
#     
# 
#     
#     plt.contour(fit(*np.indices(data.shape)), cmap=plt.cm.copper)
#     ax = plt.gca()
#     (height, x, y, width_x, width_y, b) = params
#     
#     
#     plt.text(0.95, 0.05, """
#     x : %.1f
#     y : %.1f
#     width_x : %.1f
#     width_y : %.1f""" %(x, y, width_x, width_y),
#             fontsize=12, horizontalalignment='right',
#             verticalalignment='bottom', transform=ax.transAxes)
#         
#     ax = plt.axes(projection='3d')
#     
# =============================================================================

    x = np.repeat(np.arange(30),30)
    y = np.tile(np.arange(30),30)
    #print(x)
    z = data
    
    X, Y, Z = (u.reshape(30, 30) for u in (x, y, z))
    
    ax = plt.axes(projection='3d')
    
    ax.plot_surface(X, Y, Z, cmap="autumn_r", lw=0.5, rstride=1, cstride=1)
    ax.contour(X, Y, Z, 10,  cmap="autumn_r", linestyles="solid", offset=-1)
    ax.contour(X, Y, Z, 10, colors="k", linestyles="solid")
    
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    
    plt.title("Graph#" + str(numero))       
    plt.savefig('3dgraphs1gaus/graph#' + str(numero) +  '.png')
                
                
                
                
    ax = plt.axes(projection='3d')
    
    difference = diffarray.copy()
    difference = difference.reshape(900,1)
    
    
    numdifference = np.std(difference)
    #numdifference = np.mean(np.absolute(difference))
    


    x = np.repeat(np.arange(30),30)
    y = np.tile(np.arange(30),30)

    z = diffarray
    
    X, Y, Z = (u.reshape(30, 30) for u in (x, y, z))
    
    ax.plot_surface(X, Y, Z, cmap="autumn_r", lw=0.5, rstride=1, cstride=1)
    ax.contour(X, Y, Z, 10,  cmap="autumn_r", linestyles="solid", offset=-1)
    ax.contour(X, Y, Z, 10, colors="k", linestyles="solid")
    
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    #return(numdifference)
    return(height, xspot, yspot)


def realData1Gvs2G():
    difference_set2 = []
    distance = []
    for i in range(len(newdata)):
        #GausData(i, newdata[i])
        
        off1, off2, width_x, width_y, width_x1, width_y1,x, y, x1, y1, height, he1 , dist = TwoGausData(i, newdata[i])
        twoheight = height + he1
        oneheight, xspot, yspot = OneGausData(i, newdata[i])
        error = 100*((twoheight - oneheight)/twoheight)
        
        if(error < 0 or error > 10):
            print("Fail Case: Number#", i , "has some issues. ", "Error: ", error,"Details:",
                   "twoheight:", twoheight, height, he1, "oneheight:", oneheight)
        difference_set2.append(error)
        distance.append(dist)
    plt.figure()
    plt.ylabel("Error(%)")
    plt.xlabel("Distance between Gaussians")
    #linepar = np.poly1d(np.polyfit(distance, difference_set2, 1))
    plt.plot(distance, difference_set2, 'bo')
    #plt.plot(distance, linepar(distance))
    plt.savefig("analysis/RealData1Gvs2G_Distance.png")
    plt.show()

def real2SpotLoc():
    #difference_set2 = []
    #distance = []
    scale = 30
    for i in range(len(newdata)):
         
        off1, off2, width_x, width_y, width_x1, width_y1, x, y, x1, y1, height, he1 , dist = TwoGausData(i, newdata[i])
        img = cv2.imread("data/image#" + str(i) + ".tif")
        img = cv2.resize(img, (30*scale, 30*scale), interpolation = cv2.INTER_NEAREST)
    
        x = int(scale * np.round(x))
        y = int(scale * np.round(y))
                
        #cv2.circle(img,(x,y), 1, (0,0,255), -1)
        cv2.line(img,(x,y),(x,y),(0,165,255), int(np.round(scale*height)))
        
        x1 = int(scale * np.round(x1))
        y1 = int(scale * np.round(y1))
        
        cv2.line(img, (x1,y1), (x1,y1),(0,255,0), int(np.round(scale*he1)))
        
        #gaussian boundaries
        width_x = int(scale * np.round(width_x))
        width_y = int(scale * np.round(width_y))
        
        cv2.ellipse(img, (x,y),(width_x, width_y), 0, 0, 360, (0,165,255), 2)
        
        width_x1 = int(scale * np.round(width_x1))
        width_y1 = int(scale * np.round(width_y1))
        
        cv2.ellipse(img, (x1,y1),(width_x1, width_y1), 0, 0, 360, (0,255,0), 2)
        
        #rcv2.putText(img,'OpenCV Tuts!',(x,y), 3, (200,255,155), 13, cv2.LINE_AA)
        path = "procdata/image#" + str(i) + ".tif"
        cv2.imwrite(path, img)
        

# height, he1, dist = TwoGausData(0, simulateGaus())
# twoheight = height + he1
# oneheight = OneGausData(0, simulateGaus())
# error = 100*((twoheight - oneheight)/twoheight)
# if(error > 100 or error < 0):
#     print(height, he1)
# else:
#     difference_set2.append(error)
#     distance.append(dist)
# 
# =============================================================================
def simTotalAmpErr(datapoints):
    
    spotloc1, spotloc2, gausarr, dist, heightreal1, heightreal2 = simulateGaus(datapoints)
    
    error = []
    for i in range(len(gausarr)):
        heightreal = heightreal1[i] + heightreal2[i]
        off1, off2, width_x, width_y, width_x1, width_y1, x, y, x1, y1, height, he1, dista = TwoGausData(i, gausarr[i])
        #Single spot error (not good)
        #err = 100 * (min((abs(heightreal[i] - height)/heightreal[i]), abs((heightreal[i] - he1)/heightreal[i])))
        #Error of combined fluorescent
        err = 100 * (heightreal - height - he1)/(heightreal)
        if(err > 40):
            plt.figure()
            plt.title("The error is " + str(err))
            print("The real height is : ", heightreal, "and the estimates are " , height, " and ", he1)     
            plt.imshow(gausarr[i])
        
        error.append(err)
    
    
    plt.figure()
    plt.ylabel("Error(%)")
    plt.xlabel("Distance between Gaussians")
    #linepar = np.poly1d(np.polyfit(dist, error, 1))
    plt.plot(dist, error, 'bo')
    #plt.plot(dist, linepar(dist ))
    plt.savefig("analysis/sim_distance_error_totalamp.png")
    plt.show()
def simSingleSpotErr(datapoints):
    
    spotloc1,spotloc2, gausarr, dist, heightreal1, heightreal2 = simulateGaus(datapoints)
    error = []
    
    for i in range(len(gausarr)):
        off1, off2, width_x, width_y, width_x1, width_y1, x, y, x1, y1, height, he1, dista = TwoGausData(i, gausarr[i])
        #Single spot error (not good)
        err = 100 * (min((abs(heightreal1[i] - height)/heightreal1[i]), abs((heightreal1[i] - he1)/heightreal1[i])))
        #Error of combined fluorescent
        #err = 100 * (heightreal1[i] - height - he1)/(heightreal1[i])
        if(err > 40):
            plt.figure()
            plt.title("The error is" + str(err))
            print("The real height is : ", heightreal1[i], "and the estimates are " , height, " and ", he1)     
            plt.imshow(gausarr[i])
        
        error.append(err)
    
    
    plt.figure()
    plt.ylabel("Error(%)")
    plt.xlabel("Distance between Gaussians")
    #linepar = np.poly1d(np.polyfit(dist, error, 1))
    plt.plot(dist, error, 'bo')
    #plt.plot(dist, linepar(dist ))
    plt.savefig("analysis/sim_distance_error_singlespot.png")
    plt.show()
   
def twoSpotLoc(datapoints):
    spotloc1, spotloc2,gausarr, dist, heightreal1, heightreal2 = simulateGaus(datapoints)
    error = []
    for i in range(len(gausarr)):
        off1, off2, width_x, width_y, width_x1, width_y1, x, y, x1, y1, height, he1, dista = TwoGausData(i, gausarr[i])
        #Single spot error (not good)
        
        diff1 = np.sqrt((spotloc1[i][0] - x)**2 + (spotloc1[i][1] - y)**2)
        diff2 = np.sqrt((spotloc2[i][0] - x1)**2 + (spotloc2[i][1] - y1)**2)
        
        diff11 = np.sqrt((spotloc1[i][0] - x1)**2 + (spotloc1[i][1] - y1)**2)
        diff21 = np.sqrt((spotloc2[i][0] - x)**2 + (spotloc2[i][1] - y)**2)
        
        
        err = min(diff1 + diff2, diff11 + diff21)
        #print(spotloc1[i][0], x, spotloc1[i][1], y)
        #print(spotloc2[i][0], x1, spotloc2[i][1], y1)
        #err = diff1 + diff2
        if(err > 5):
            plt.figure()
            
            plt.title("The error is " + str(err))
            
            plt.imshow(gausarr[i])
            plt.savefig("data/error#" + str(i))
            print(i ," is the error case with error of ", err)
            print("Details: 1st spot", spotloc1[i][0], x, spotloc1[i][1], y) 
            print("Details: 2nd spot", spotloc2[i][0], x1, spotloc2[i][1], y1)
        #print(np.shape(x))
        #Error of combined fluorescent
        #err = 100 * (heightreal1[i] - height - he1)/(heightreal1[i])
        error.append(err)
    plt.figure()
    plt.ylabel("Error in Pixel distance (of both spots predicted vs. real)")
    plt.xlabel("Distance between Gaussians")
    #linepar = np.poly1d(np.polyfit(dist, error, 1))

    plt.plot(dist, error, 'bo')
    #plt.plot(dist, linepar(dist ))
    plt.savefig("analysis/sim_distance_error_singlespot.png")
    plt.show()

def singleSpotLoc(datapoints):
    spotloc1, gausarr, heightreal1 = simulateOneGaus(datapoints)
    error = []
    for i in range(len(gausarr)):
        height, xspot, yspot =  OneGausData(i, gausarr[i])
        diff1 = np.sqrt((spotloc1[i][0] - xspot)**2 + (spotloc1[i][1] - yspot)**2)
        print(spotloc1[i][0], xspot, spotloc1[i][1], yspot)
        err = diff1
        error.append(err)
             
        plt.figure()
        plt.title("The error is" + str(err))
        print()
    #print("The real height is : ", heightreal1[i], "and the estimates are " , height, " and ", he1)     
        plt.imshow(gausarr[i])
    #plt.scatter(np.round(spotloc1[i][0]), np.round(spotloc1[i][1]))
    #plt.scatter(xspot,yspot)
# =============================================================================
    plt.figure()
    plt.ylabel("Error in Pixel distance (of simulated center and predicted center)")
    plt.xlabel("Test cases")
    #linepar = np.poly1d(np.polyfit(dist, error, 1))
    plt.plot(np.arange(len(error)), error, 'bo')
    #plt.plot(dist, linepar(dist ))
    plt.savefig("analysis/sim_distance_error_singlespot.png")
    
    plt.show()

real2SpotLoc()
print("completed")


# plt.figure()
    # plt.ylabel("Error(%)")
# plt.xlabel("Distance between Gaussians")
# linepar = np.poly1d(np.polyfit(distance, difference_set2, 1))
# plt.plot(distance, difference_set2, 'bo')
# plt.plot(distance, linepar(distance))
# plt.savefig("analysis/distance_error_2dgaus_rel.png")
# plt.show()

# =============================================================================
# plt.figure()
# print(difference_set2)
# difference_set2 = np.array(difference_set2)
# num_bins = 5
# n, bins, patches = plt.hist(difference_set2, num_bins, facecolor='blue', alpha=0.5)
# 
# mean = plt.axvline(np.mean(difference_set2), color='red', linestyle='dashed', linewidth=1)
# 
# mean.set_label("mean")
# plt.legend()
# 
# plt.xlabel("Difference")
# #Standard Deviation
# plt.ylabel("Occurrences")
# plt.title("Error")
# #Average Standard deviation difference between Gaussian Fit and Data
# 
# plt.savefig("analysis/rihjt.png")
# 
# plt.show()
# =============================================================================

    
#root mean square deviation
#gaussian mixture model package
#polyfit


# =============================================================================
# print(difference_set)
# plt.plot(np.arange(len(difference_set)), difference_set)
# plt.ylabel("Standard Deviation")
# 

# =============================================================================
# def twoD_Gaussian(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
#     (x,y) = xdata_tuple
#     #print(x)
#     #print(y)
#     xo = float(xo)
#     yo = float(yo)   
#     #amplitude = testdata[y,x]
#     #print(testdata[x,y])
#     
#     a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
#     b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
#     c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
#     g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
#                             + c*((y-yo)**2)))
#     return g.ravel()
# 
# 
# x = np.tile(np.arange(30), 30)
# y = np.repeat(np.arange(30), 30)
# #x, y = np.meshgrid(x, y)
# amp = testdata[y,x]
# #create data
# print(np.shape((x, y)))
# data = twoD_Gaussian((x, y), 1.0, 15, 15, 2, 2, 0, 0)
# 
# #print(data)
# 
# # plot twoD_Gaussian data generated above
# plt.figure()
# plt.imshow(data.reshape(30, 30))
# plt.colorbar()
# plt.title("here")
# initial_guess = (3,100,100,20,40,0,0)
# 
# data_noisy = data + 0.00*np.random.normal(size=data.shape)
# =============================================================================



#3D plot ->


# =============================================================================
# 
# popt, pcov = opt.curve_fit(twoD_Gaussian,(x,y),data,  bounds= (0.01, [30,30,1,10,10,1,1]))
# 
# print(pcov)
# print(popt)
# data_fitted = twoD_Gaussian((x, y), *popt)
# 
# 
# fig, ax = plt.subplots(1, 1)
# plt.title("Next One")
# ax.imshow(data_fitted.reshape(30, 30), cmap=plt.cm.jet, origin='bottom',
#     extent=(x.min(), x.max(), y.min(), y.max()))
# #ax.contour(x, y, data_fitted.reshape(30, 30),8, colors='w')
# plt.show()
# 
# =============================================================================

# =============================================================================
# def gaussian_2d(xy_mesh, amp, xc, yc, sigma_x, sigma_y):
#     
#     (x, y) = xy_mesh
#     amp = testdata[y,x]
# 
#     gauss = amp*np.exp(-((x-xc)**2/(2*sigma_x**2)+(y-yc)**2/(2*sigma_y**2)))/(2*np.pi*sigma_x*sigma_y)
#     
#     #2d into 1d
#     return np.ravel(gauss)
# x = np.arange(30)
# y = np.arange(30)
# xy_mesh = np.meshgrid(x,y)
# 
# amp = 1
# xc, yc = np.median(x), np.median(y)
# sigma_x, sigma_y = x[-1]/10, y[-1]/6
#  
# noise_factor = 0.02
#  
# # make both clean and noisy data, reshaping the Gaussian to proper 2D dimensions
# data = gaussian_2d(xy_mesh, amp, xc, yc, sigma_x, sigma_y).reshape(np.outer(x, y).shape)
# noise = data + noise_factor*data.max()*np.random.normal(size=data.shape)
# 
# 
# # plot the function and with noise added
# plt.figure(figsize=(8,4))
# plt.subplot(1,2,1)
# plt.title('model')
# plt.imshow(data, origin='bottom')
# plt.grid(visible=False)
# plt.subplot(1,2,2)
# plt.title('noisy data')
# plt.imshow(noise, origin='bottom')
# plt.grid(visible=False)
# plt.show()
# =============================================================================



# =============================================================================
# def gaus(x,a,x0,sigma):
#     return a*exp(-(x-x0)**2/(2*sigma**2))
#     return a*exp(-(x-x0)**2/(sigma**2))
# fitting = testdata[10]
# def oneDGausDisp(number, arr):
#     x = ar(range(2 * boxsize))
#     y = ar(arr)
#                          
#     mean = np.mean(x)          
#     sigma = np.std(x)     
# 
#     popt,pcov = curve_fit(gaus,x,y,p0=[1,mean,sigma], maxfev=50000000)
#     plt.figure(number + 1)
#     plt.plot(x,y,'bo:',label='data')
#     plt.plot(x,gaus(x,*popt),'ro:',label='fit')
#     plt.legend()
#     print(gaus(x,*popt))
# 
# def oneDGaus(arr):
#     x = ar(range(2 * boxsize))
#     y = ar(arr)
#                          
#     mean = np.mean(x)          
#     sigma = np.std(x)     
# 
#     popt,pcov = curve_fit(gaus,x,y, p0 = [1,mean,sigma], maxfev=500000000)
# 
#     print(gaus(x, *popt))
#     print(popt)
# 
# oneDGausDisp(9, testdata[20])
# =============================================================================
#for i in range(len(testdata)):
#    print("Graph#" , i)
#    oneDGausDisp(i, testdata[i])
    


#ppc = pm.sample_posterior_predictive(trace, samples=500, model=model)

#print(np.asarray(ppc['n']).shape)

#_, ax = plt.subplots(figsize=(12, 6))
#ax.hist([n.mean() for n in ppc['n']], bins=19, alpha=0.5)
#ax.axvline(testdata[15].mean())
#ax.set(title='Posterior predictive of the mean', xlabel='mean(x)', ylabel='Frequency');
#pm.plots.traceplot(obs)



#samp = norm.rvs(loc=0,scale=1,size=150) 

#param = norm.fit(samp) # distribution fitting

# now, param[0] and param[1] are the mean and 
# the standard deviation of the fitted distribution
#x = linspace(-5,5,100)
# fitted distribution
#pdf_fitted = norm.pdf(x,loc=param[0],scale=param[1])
# original distribution
#pdf = norm.pdf(x)

#title('Normal distribution')
#plot(x,pdf_fitted,'r-',x,pdf,'b-')
#hist(samp,density=1,alpha=.3)
#show()


#cov = np.array(data[0])
#mu = np.zeros(30)
#vals = pm.MvNormal('vals', mu=mu, cov=cov, shape=(5, 2))


#for keypoint in keypoints:
#    im[keypoint.pt[0], keypoint.pt[1]] = [0,255,0]     