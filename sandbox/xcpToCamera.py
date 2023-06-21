# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 13:32:13 2023

@author: suhlr
"""

import os
import numpy as np
import xmltodict
import copy
import sys
import cv2
import scipy.spatial.transform as spTransform

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import utils

def xcpToCamera(xcpPath,cameraIDs='Vue'):
    # CameraIDs is a dict of ids, or a type of camera, which will use all cameras
    # of this type.
    
    # parse xcp file
    with open(xcpPath) as fd:
        xmlDict = xmltodict.parse(fd.read())
    
    # if camera type specified, find cameras ids
    if not isinstance(cameraIDs, list):
        cameraType = copy.copy(cameraIDs)
        cameraIDs = [c['@DEVICEID'] for c in xmlDict['Cameras']['Camera'] if c['@DISPLAY_TYPE'] == cameraType]
    
    # XML parameters come in as a list of numbers
    def stringToList(string):
        thisList = string.split(' ')
        
        for i,s in enumerate(thisList):
            try:
                thisList[i] = float(s)
            except:
                pass        
        return thisList
    
    # extract parameters
    for camID in cameraIDs:
        cameraParameters = {}
        # Parameters defined here https://www.researchgate.net/publication/347567947_Towards_end-to-end_training_of_proposal-based_3D_human_pose_estimation
        thisCam = next(cam for cam in xmlDict['Cameras']['Camera'] if cam["@DEVICEID"] == camID)
        # image size
        sensorSize = stringToList(thisCam['@SENSOR_SIZE'])
        cameraParameters['imageSize'] = np.expand_dims(np.array((sensorSize)),axis=1)
        
        # vicon only uses 3 distortion parameters for r^2, r^4, r^6. Set the final two to 0 for opencv undistort (denominator)
        distortion = stringToList(thisCam['KeyFrames']['KeyFrame']['@VICON_RADIAL2'])
        cameraParameters['distortion'] = np.expand_dims(np.array(distortion[3:6] + [0,0]),axis=0)

        # intrinsic matrix 
        principalPoint = stringToList(thisCam['KeyFrames']['KeyFrame']['@PRINCIPAL_POINT'])
        focalLength = stringToList(thisCam['KeyFrames']['KeyFrame']['@FOCAL_LENGTH'])[0]
        cameraParameters['intrinsicMat'] = np.diag(np.array([focalLength, focalLength, 1]))
        cameraParameters['intrinsicMat'][0,2] = principalPoint[0]
        cameraParameters['intrinsicMat'][1,2] = principalPoint[1]
        
        # rotation 
        quat = stringToList(thisCam['KeyFrames']['KeyFrame']['@ORIENTATION'])
        cameraParameters['rotation'] = spTransform.Rotation.as_matrix((spTransform.Rotation.from_quat(quat)))
        
        # rotation_EulerAngles 3x1
        cameraParameters['rotation_EulerAngles'] = np.array(cv2.Rodrigues(cameraParameters['rotation'])[0])
        
        # translation 3x1
        translation = stringToList(thisCam['KeyFrames']['KeyFrame']['@POSITION'])
        cameraParameters['translation'] = np.expand_dims(np.array(translation),axis=1)

    return cameraParameters  

# %% testing as script
xcpPath = os.path.join('C:/Users/hpl/Documents/MyRepositories/opencap-core_ACL/sandbox',
                       'testCalibration.xcp')

xcpToCamera(xcpPath)

test = 1

def stringToList(string):
    return string.split(' ')    

test = 'this 10 50'

out = stringToList(test)


