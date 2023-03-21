# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 01:14:29 2023

@author: suhlr
"""

import utilsMocap
import os
import cv2
import utils
import utilsChecker
import numpy as np

basePath = 'C:/SharedGdrive/HPL_MASPL/OpenCap/Data/S1003/'
cameraParametersPath = os.path.join(basePath,'Videos','Cam1',
                                    'cameraIntrinsicsExtrinsics.pickle')
videoPath = os.path.join(basePath,'Videos','Cam1','InputMedia','S1003_DLSQUATS','S1003_DLSQUATS_rotated.avi')
imagePath = os.path.join(os.path.dirname(videoPath),'output0.png')

# pop an image
utilsChecker.video2Images(videoPath, nImages=2, tSingleImage=.1, 
                          filePrefix='output', outputFolder='default')

out = utilsMocap.pickPointsFromImage(imagePath,6)

points2d, points3d = utilsMocap.sampleGrid(out, gridDims = [(-40,40),(0,60),(0,0)])

# load in some camera params for intrinsics
CameraParams = utils.loadCameraParameters(cameraParametersPath)

# draw 2d points on image
points2dForPlotting = [[int(p[0]), int(p[1])] for p in points2d]

image = cv2.imread(imagePath)
for i in range(len(points2d)):
    image = cv2.circle(image, tuple((points2dForPlotting[i])), radius=2, color=(0, 255, 0), thickness=-1)
cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()









# This function gives two possible solutions.
   # It helps with the ambiguous cases with small checkerboards (appears like
   # left handed coord system). Unfortunately, there isn't a clear way to 
   # choose the correct solution. It is the nature of the solvePnP problem 
   # with a bit of 2D point noise.
rets, rvecs, tvecs, reprojError = cv2.solvePnPGeneric(
    points3d, points2d, CameraParams['intrinsicMat'], 
    CameraParams['distortion'], flags=cv2.SOLVEPNP_IPPE)
rvec = rvecs[1]
tvec = tvecs[1]
  
if rets < 1 or np.max(rvec) == 0 or np.max(tvec) == 0:
    print('solvePnPGeneric failed. Use SolvePnPRansac')
    # Note: can input extrinsics guess if we generally know where they are.
    # Add to lists to look like solvePnPRansac results
    rvecs = []
    tvecs = []
    ret, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectp3d, corners2, CameraParams['intrinsicMat'],
        CameraParams['distortion'])

R_worldFromCamera = cv2.Rodrigues(rvec)[0]
    
theseCameraParams['rotation'] = R_worldFromCamera
theseCameraParams['translation'] = tvec
theseCameraParams['rotation_EulerAngles'] = rvec

# save extrinsics parameters to video folder
# will save the selected parameters in Camera folder in main
saveExtPath = os.path.join(
    os.path.dirname(imageFileName),
    'cameraIntrinsicsExtrinsics_soln{}.pickle'.format(iRet))
saveCameraParameters(saveExtPath,theseCameraParams)