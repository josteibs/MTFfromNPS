# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 15:33:49 2023

@author: Jostein B. Steffensen
MTF from NPS
"""

import os
import PySimpleGUI as sg

from image_calc import NPS_xray, MTFfromNPS

working_directory = 'F:/Røntgen/Arbeidsmappe/2023/2023 MTF fra NPS' #os.getcwd()

### Window design
layout = [
    # import image series
    [sg.Text("Import image series for linear regression:")],
    [sg.InputText(key='-FOLDER_PATH-', enable_events=True),
     sg.FolderBrowse(initial_folder=working_directory)],
    # View ROI
    [sg.Button("Check ROI", key='-ROI-', disabled=True)],
    [sg.Button("2D NPS", key='-2DNPS-', disabled=True)],
    [sg.Button("NPS", key='-NPS-', disabled=True)]
]
window = sg.Window("MTF from NPS", layout, size=((550,300)))


### GUI action
while True:
    event, values = window.read()
    
    if event=='-FOLDER_PATH-':
        window['-ROI-'].update(disabled=False)
        window['-2DNPS-'].update(disabled=False)
        window['-NPS-'].update(disabled=False)
        
        path = values['-FOLDER_PATH-'] + '/*'
        images = MTFfromNPS(path)
        images.setROI(600, 600, 100)
        images.image_dict()
        
        # NPS calculations
        # images.ROIcube()
        # images.fft2_avg()
        
    if event == '-ROI-':
        
        images.showROI()
            
    if event == '-2DNPS-':
        images.NPS_2D()
            
    if event == '-NPS-':
        images.radial_avg()
    
    if event == sg.WIN_CLOSED:
        break
    
window.close()