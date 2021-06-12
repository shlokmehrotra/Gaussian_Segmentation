#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 14:51:38 2019

@author: shlokmehrotra
"""
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
from scipy.stats import multivariate_normal

def threeGausData(numero, theData):
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
        
    #this should work
    def gaussian_norot(height, center_x, center_y, width_x, width_y, height1, center_x1, center_y1, width_x1, width_y1):
        """Returns a gaussian function with the given parameters"""
        width_x = float(width_x)
        width_y = float(width_y)
        width_x1 = float(width_x1)
        width_y1 = float(width_y1)
        return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2) + height1*np.exp(
                -(((center_x1-x)/width_x1)**2+((center_y1-y)/width_y1)**2)/2)
    #problematic
    def moments(data):
        """Returns  (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution by calculating its
        moments """
        return(1, 15, 14, 3, 3, 1, 14, 16, 3, 2)
    
    
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
        height, x, y, width_x, width_y, he1, x1,y1, wi1, wi2  = params
        #p, success = optimize.leastsq(errorfunction, params)
        bnds = (0,30)
        p = optimize.least_squares(errorfunction, params, bounds = bnds).x
        
        #least square fitting(minimizes raw data and fit)

        if(p[0] < 1 and p[5] < 1 and p[1] > 0 and p[1] < 30 and p[2] > 0 and p[2] < 30 and p[6] > 0 and p[6] < 30 and p[7] > 0 and p[7] < 30):
             #print("pass case")
             return(p)
        else:
            print("failed case")
            print("height1", p[0],"height2", p[5], "X",  p[1],"Y",  p[2],"Y1",  p[6], "Y2", p[7])
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
    (height, xspot, yspot, width_x, width_y, he1, x1spot,y1spot, wi1, wi2) = params
    distance = (np.sqrt((xspot - x1spot)**2 + (yspot - y1spot)**2))

    ax = plt.axes(projection='3d')
    
    
    
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
                
    return(xspot, yspot, x1spot, y1spot, height, he1, distance)
threeGausData()

