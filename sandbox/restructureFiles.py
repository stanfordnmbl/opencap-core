# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 14:14:17 2023

@author: Scott Uhlrich
"""

import os
import glob
import shutil

def moveFilesToOpenCapStructure(inputDirectory,outputDirectory,camList): 
    # find all avi files of a certain name in a directory
    files = glob.glob(os.path.join(inputDirectory, '*.avi'))
    trialNames = [os.path.split(string[:string.find('.')])[1] for string in files]
    trialNames = list(set(trialNames))

    # create file structure
    for i,_ in enumerate(camList):
        os.makedirs(os.path.join(outputDirectory, 'Cam' + str(i),'InputMedia'),exist_ok=True)

    # copy files to new structure
    for i,trial in enumerate(trialNames):
        for j,cam in enumerate(camList):
            # make subfolder of trial name
            os.makedirs(os.path.join(outputDirectory, 'Cam' + str(j),'InputMedia', trial),exist_ok=True)
            # copy file to subfolder
            shutil.copyfile(glob.glob(os.path.join(inputDirectory,trial + '.' + cam + '*.avi'))[0],
                                    os.path.join(outputDirectory, 'Cam' + str(j),'InputMedia', trial , trial + '.avi'))
            
# %% testing as script
inputDirectory = 'G:/Shared drives/HPL_MASPL/DATA/S1003/S1003'
outputDirectory = 'G:/Shared drives/HPL_MASPL/DATA/S1003/S1003/OpenCap'
camList = ['2122194','2141511','2111275']

moveFilesToOpenCapStructure(inputDirectory,outputDirectory,camList)

