# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 10:51:56 2023

@author: josste
"""
import glob
import pydicom
from matplotlib import rc

def sortImages(pathname):
    '''Function from Vilde
    Sort images in same directory'''
    sortDict = {}
    for path in glob.glob(pathname):
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        sortDict[ds.SliceLocation] = path
        rc('figure', max_open_warning = 0)
    sortedKeys = sorted(sortDict.keys())
    return sortDict, sortedKeys