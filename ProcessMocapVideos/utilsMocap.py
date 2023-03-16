# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 21:39:08 2023

@author: Scott Uhlrich
"""

import os
import yaml
import glob
import shutil
import numpy as np
import xmltodict
import copy
import sys
import cv2
import scipy.spatial.transform as spTransform

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import utils
import utilsChecker

# %% 
def moveFilesToOpenCapStructure(inputDirectory,outputDirectory,camList=None,calibration=False,copyVideos=True): 
    # find all avi files of a certain name in a directory
    files = glob.glob(os.path.join(inputDirectory, '*.avi'))
    trialNames = [os.path.split(string[:string.find('.')])[1] for string in files]
    trialNames = list(set(trialNames))
    
    # get camList from first trial
    if camList is None:
        trials = [os.path.split(s)[1] for s in files if trialNames[0] in s]
        camList = []
        for t in trials:
            t = t[t.find('.')+1:]            
            camList.append(t[:t.find('.')])  

    # create file structure
    for i,_ in enumerate(camList):
        os.makedirs(os.path.join(outputDirectory,'Videos', 'Cam' + str(i),'InputMedia'),exist_ok=True)

    # copy files to new structure
    for i,trial in enumerate(trialNames):
        for j,cam in enumerate(camList):
            # make subfolder of trial name
            os.makedirs(os.path.join(outputDirectory,'Videos', 'Cam' + str(j),'InputMedia', trial),exist_ok=True)
            # copy file to subfolder
            if copyVideos:
                shutil.copyfile(glob.glob(os.path.join(inputDirectory,trial + '.' + cam + '*.avi'))[0],
                                    os.path.join(outputDirectory,'Videos', 'Cam' + str(j),'InputMedia', trial , trial + '.avi'))
    
    # write a yaml that maps camList to 'Cam0', 'Cam1', etc.
    camDict = {}
    for i,cam in enumerate(camList):
        camDict['Cam' + str(i)] = cam
    with open(os.path.join(outputDirectory,'Videos','mappingCameras.yaml'), 'w') as file:
        documents = yaml.dump(camDict, file)
            
    if calibration:
        # find first calibration, and put in the camera folders
        inCalibrationPath = glob.glob(os.path.join(inputDirectory,'*.xcp'))[0]
        os.makedirs(os.path.join(outputDirectory,'Calibration'),exist_ok=True)
        outCalibrationPath = os.path.join(outputDirectory,'Calibration','viconCalibration.xcp')
        shutil.copyfile(inCalibrationPath,outCalibrationPath)
        
        # Convert calibrations and save
        xcpToCameraParameters(outCalibrationPath,camList,
                              saveBasePath=os.path.join(outputDirectory,'Videos'))
               
    return

# %% Create Metadata

def createMetadata(outputDirectory, height=1.7, mass=75, subjectID='default'):
    repoDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sessionMetadata = utils.importMetadata(os.path.join(repoDir,'defaultSessionMetadata.yaml'))

    # change session_desc to sessionMetadata   
    sessionMetadata["subjectID"] = subjectID
    sessionMetadata["mass_kg"] = float(mass)
    sessionMetadata["height_m"] = float(height)
    
    sessionYamlPath = os.path.join(outputDirectory,'sessionMetadata.yaml')
    with open(sessionYamlPath, 'w') as file:
        yaml.dump(sessionMetadata, file)
    
    return

# %% 
def xcpToCameraParameters(xcpPath,cameraIDs='Vue',saveBasePath=None):
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
    for i,camID in enumerate(cameraIDs):
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
        
        if saveBasePath is not None:
            utilsChecker.saveCameraParameters(os.path.join(saveBasePath,'Cam' + str(i),
                                                           'cameraIntrinsicsExtrinsics.pickle'),
                                                           cameraParameters)

    return cameraParameters  
