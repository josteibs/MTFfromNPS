# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 12:24:42 2023

@author: Jostein Brunvær Steffensen
Class containing NPS calculations 
"""
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import pydicom 
import scipy.fft as fft
import time


def sortImages(pathname):
    '''Function from Vilde
    Sort images in same directory'''
    sortDict = {}
    for path in glob.glob(pathname):
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        sortDict[ds[0x018, 0x1153].value] = path
        rc('figure', max_open_warning = 0)
    sortedKeys = sorted(sortDict.keys())
    return sortDict, sortedKeys

### Image class: Containing image stats and ROIs
class Image:
    def __init__(self, image_array, ROI_array):
        self.image = image_array
        self.ROI_array = ROI_array # 2D if only containing one ROI. 3D else.        

### Class aiming to calculate MTF from NPS. 
class MTFfromNPS:
    def __init__(self, folder_path):
        self.sortDict, self.sortedKeys = sortImages(folder_path) #Sort images
        
        # Using first image to display ROI
        data0 = self.sortDict[self.sortedKeys[0]]
        self.dicom0 = pydicom.dcmread(data0)
        
    def setROI(self, x_up_left_corner, y_up_left_corner, ROIsize):
        self.x_up_left_corner = x_up_left_corner
        self.y_up_left_corner = y_up_left_corner
        self.ROIsize = ROIsize
    
    # Display first image with ROI
    def showROI(self):
        image0 = self.dicom0.pixel_array
        fig, ax = plt.subplots()
        im = ax.imshow(image0, cmap='Greys_r', vmin=np.min(image0), vmax=np.max(image0))
        
        # Marking ROI
        rect = mpl.patches.Rectangle((self.x_up_left_corner, self.y_up_left_corner), self.ROIsize, self.ROIsize, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.show()
        
        #2D match
        #plt.figure()
        #plt.imshow(surface_fit(image0[self.ROIcorner:self.ROIcorner+self.ROIsize, self.ROIcorner:self.ROIcorner+self.ROIsize]))
        #plt.show()
    
    # Making 3d array of ROI
    def image_dict(self):
        self.image_dict = {}
        start_time = time.time()
        for key in self.sortedKeys:
            data = self.sortDict[key]
            dicom = pydicom.dcmread(data) # loading this is what makes the program slow. 
            image = dicom.pixel_array
            ROI = image[self.x_up_left_corner:self.x_up_left_corner+self.ROIsize, self.y_up_left_corner:self.y_up_left_corner+self.ROIsize]
            # Making image object
            image_obj = Image(image, ROI)
            # Add to image_dict. Key is the same.
            self.image_dict[key] = image_obj
        print(f'# Images: {len(self.image_dict)}')
        
    def NPS_2D(self):
        self.nps_obj = NPS_xray(self.image_dict[self.sortedKeys[0]].ROI_array)
        self.nps_obj.fft2_avg(0.1)
        self.nps_obj.NPS_2D_show()
        
    def radial_avg(self):
        self.nps_obj.radial_avg()
        
### Class calculating NPS for an x-ray image 
class NPS_xray:
    def __init__(self, ROI_cube):
       self.ROI_cube = ROI_cube
       self.ROIsize = ROI_cube.shape[0]
       try:
           self.ROI_number = len(ROI_cube[0,0,:])
       except:
           self.ROI_number = 1
    # Calulating 2D NPS
    def fft2_avg(self, pix):
        # frame for summing fft images
        self.averageROI_fft = np.zeros((self.ROIsize, self.ROIsize))
        # summing fft of ROIs
        for i in range(self.ROI_number):
            #FFT of subtracted ROI
            ROI_sub = self.ROI_cube - np.mean(self.ROI_cube)
            ROI_fft = fft.fft2(ROI_sub)
            ROI_fft_mod2 = (np.real(ROI_fft)**2 + np.imag(ROI_fft)**2)*(pix**2)/(self.ROIsize**2)
            self.averageROI_fft += ROI_fft_mod2
        self.averageROI_fft = fft.fftshift(self.averageROI_fft/self.ROI_number)
    
    # Displaying 2D NPS
    def NPS_2D_show(self):
        plt.figure()
        plt.imshow(self.averageROI_fft, cmap='Greys_r', vmin=min(self.averageROI_fft[50,:]), vmax=max(self.averageROI_fft[50,:]))
        plt.show()
      
    # Radial average of 2D NPS    
    def radial_avg(self):
        # Creds to Naveen Venkatesan for the idea for this function
        cen_x = self.ROIsize//2
        cen_y = self.ROIsize//2
        print(cen_x)
        print(cen_y)
        
        # Find radial distances 
        [X, Y] = np.meshgrid(np.arange(self.ROIsize)-cen_x, np.arange(self.ROIsize)-cen_y)
        R = np.sqrt(np.square(X)+np.square(Y))
        
        rad = np.arange(1, np.max(R), 1)
        intensity = np.zeros(len(rad))
        index = 0
        bin_size = 1
        
        for i in rad:
            mask = (np.greater(R, i - bin_size) & np.less(R, i + bin_size))
            rad_values = self.averageROI_fft[mask]
            intensity[index] = np.mean(rad_values)
            index += 1
            
        # Create figure and add subplot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # Plot data
        ax.plot(rad, intensity, linewidth=2)
        # Edit axis labels
        ax.set_xlabel('Radial Distance', labelpad=10)
        ax.set_ylabel('Average Intensity', labelpad=10)
    
    

