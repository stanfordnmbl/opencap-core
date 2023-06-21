# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 01:14:29 2023

Manually pick points of known dimensions for extrinsic calibration.

@author: suhlr
"""

import os
import utilsMocap

# We will save the calibration to a general Calibration folder by camNum
basePath = 'C:/SharedGdrive/HPL_MASPL/OpenCap/Data/S1003/'
camNum= '2111275'

# video to use for calibration. We pop an image at .1s in 
videoPath = os.path.join(basePath,'Videos','Cam1','InputMedia',
                         'S1003_DLSQUATS','S1003_DLSQUATS_rotated.avi')

# where are existing camera parameters. Only need intrinsics, but ok if everything.
cameraParametersPath = os.path.join(basePath,'Videos','Cam1',
                                    'cameraIntrinsicsExtrinsics.pickle')

savePaths = [os.path.join(basePath,'Calibration',
                          'manualCameraParameters_' + camNum + '.pickle'),
             cameraParametersPath]

# 3d dimensions of grid. We only support picking 6 points for now. From smallest
# x,y,z, values, along x, then along y. This is quite hardcoded in utilsMocap.sampleGrid.
# For example, we do 3 points across x, and only 2 across y.
# 0,0,0 is desired lab origin.
gridDims =[(-600,600),(0,900),(0,0)]

# Run calibration
CamParams = utilsMocap.manualExtrinsicCalibration(videoPath,gridDims,
                                                  cameraParametersPath,
                                                  savePaths)

