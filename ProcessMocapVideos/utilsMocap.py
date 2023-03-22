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
import win32api
import win32con


# See if we need to delete any
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import utils
import utilsChecker

# %% 
def moveFilesToOpenCapStructure(inputDirectory,outputDirectory,camList=None,
                                saveAnimation=False,calibration=False,copyVideos=True): 
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
                              saveBasePath=os.path.join(outputDirectory,'Videos'),
                              visualizeCameras=True,saveAnimation=saveAnimation)        
               
    return

# %% Create Metadata

def createMetadata(outputDirectory, height=1.7, mass=75, subjectID='default'):
    repoDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sessionMetadata = utils.importMetadata(os.path.join(repoDir,'defaultSessionMetadata.yaml'))

    # change session_desc to sessionMetadata   
    sessionMetadata["subjectID"] = subjectID
    sessionMetadata["mass_kg"] = float(mass)
    sessionMetadata["height_m"] = float(height)
    sessionMetadata["openSimModel"] = 'LaiArnoldModified2017_poly_withArms_weldHand_ACL'  
    
    sessionYamlPath = os.path.join(outputDirectory,'sessionMetadata.yaml')
    with open(sessionYamlPath, 'w') as file:
        yaml.dump(sessionMetadata, file)
    
    return

# %% 
def xcpToCameraParameters(xcpPath,cameraIDs='Vue',saveBasePath=None,visualizeCameras=False,saveAnimation=False):
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
    cameraParametersAll = []
    for i,camID in enumerate(cameraIDs):
        cameraParameters = {}
        
        # first check if we have manually saved camera parameters
        manualPath = os.path.join(os.path.dirname(xcpPath),'manualCameraParameters_' + camID + '.pickle')
        if os.path.exists(manualPath):
            cameraParameters = utils.loadCameraParameters(manualPath)
            print('Loaded manual calibration for Vicon camera ' + camID + ".")
            
        else: # pull from Vicon xcp          
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
            cameraParameters['intrinsicMat'][0,2] = 1920-principalPoint[0]
            cameraParameters['intrinsicMat'][1,2] = 1080-principalPoint[1]
            
            # rotation 
            quat = stringToList(thisCam['KeyFrames']['KeyFrame']['@ORIENTATION'])
            cameraParameters['rotation'] = spTransform.Rotation.as_matrix((spTransform.Rotation.from_quat(quat)))
            R_camera_to_world = cameraParameters['rotation']
            
            # rotation_EulerAngles 3x1
            cameraParameters['rotation_EulerAngles'] = np.array(cv2.Rodrigues(cameraParameters['rotation'])[0])
            
            # translation. Vicon defines this as translation from origin to camera in world,
            # our camera model wants it expressed in camera to origin in camera
            translation = stringToList(thisCam['KeyFrames']['KeyFrame']['@POSITION'])
            translation = np.expand_dims(np.array(translation),axis=1)
            cameraParameters['translation'] = -np.matmul(R_camera_to_world,translation)
        
        if saveBasePath is not None:
            utilsChecker.saveCameraParameters(os.path.join(saveBasePath,'Cam' + str(i),
                                                           'cameraIntrinsicsExtrinsics.pickle'),
                                                           cameraParameters)
        cameraParametersAll.append(cameraParameters)
    
    if visualizeCameras:
        if saveAnimation:
            saveAnimationPath = os.path.join(os.path.dirname(xcpPath),'cameraVolume.gif')
        else:
            saveAnimationPath=None
        plotCameras(cameraParametersAll,saveAnimationPath = saveAnimationPath)

    return cameraParametersAll  

# %% 3D plot of cameras in the world

def plotCameras(CameraParameters,saveAnimationPath):

    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
            self._verts3d = xs, ys, zs
    
        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            FancyArrowPatch.draw(self, renderer)
    
    def plotRt(R, t, thisAx, arrow_prop_dict):
        # plot coordinate system rotated by R and translated by t
        a = Arrow3D([t[0], t[0] + R[0,0]],
                    [t[1], t[1] + R[0,1]],
                    [t[2], t[2] + R[0,2]], **arrow_prop_dict, color='r')

        thisAx.add_artist(a)
        a = Arrow3D([t[0], t[0] + R[1,0]],
                    [t[1], t[1] + R[1,1]],
                    [t[2], t[2] + R[1,2]], **arrow_prop_dict, color='g')
        thisAx.add_artist(a)

        a = Arrow3D([t[0], t[0] + R[2,0]],
                    [t[1], t[1] + R[2,1]],
                    [t[2], t[2] + R[2,2]], **arrow_prop_dict, color='b')
        thisAx.add_artist(a)


    fig = plt.figure()
    plt.figure(figsize=(10,6))
    ax1 = fig.add_subplot(111, projection='3d')
    arrow_prop_dict = dict(mutation_scale=20, arrowstyle='->', shrinkA=0, shrinkB=0)

    # world coord frame
    plotRt(np.eye(3), [0,0,0], ax1, arrow_prop_dict)
    ax1.text(0,0,0, 'lab')



    for i,cam in enumerate(CameraParameters):
        # plot camera
        R = cam['rotation'] # R_camera_to_world
        t = cam['translation'] # t_camera_to_world]camera
        t = -np.matmul(R.T,t)/1000 # t_world_to_camera]world in meters
        t =  [tr[0] for tr in t] 

        plotRt(R, t, ax1, arrow_prop_dict)
        
        # add text with Cam # and coordinates of t in world frame rounded to one decimal
        ax1.text(t[0], t[1], t[2], 'Cam' + str(i))           

    ax1.axes.set_xlim3d(left=-5, right=5) 
    ax1.axes.set_ylim3d(bottom=-5, top=5) 
    ax1.axes.set_zlim3d(bottom=-5, top=5)

    if saveAnimationPath is not None:    
        print('Saving animation of camera setup. This is slow, disable if not needed.')
        # animate rotation of 3d projection
        def animate(i):
            ax1.view_init(elev=15, azim=i*4)
            return fig,
    
        # Init only required for blitting to give a clean slate.
        def init():
            ax1.view_init(elev=15, azim=0)
            return fig,
    
        # Animate
        anim = mpl.animation.FuncAnimation(fig, animate,
                                       frames=90, interval=5, blit=False)  
        
        # writing camera setup video
        
        writervideo = mpl.animation.FFMpegWriter(fps=10)
        anim.save(saveAnimationPath, writer=writervideo)
        
        plt.close()
        
        return

    plt.show()
    
# %% Pick points from an image for calibration

def pickPointsFromImage(imagePath,nPts):
    
    refPts = []
    def click_and_save(event, x, y, flags, param):  
        # grab references to the global variables
        nonlocal refPts
        if event == cv2.EVENT_LBUTTONDOWN:
            refPts.append((x,y))
        if event == cv2.EVENT_MOUSEMOVE:
            win32api.SetCursor(win32api.LoadCursor(0, win32con.IDC_ARROW))
    # load the image, clone it, and setup the mouse callback function
    refPts = []

    image = cv2.imread(imagePath)
    clone = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_save)
    print('Click on ' + str(nPts) + ' points or press q when finished. Smallest x/y vals to largest, cross over for top.')
    
    # keep looping until the 'q' key is pressed
    ptLength = 0
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF
        
        if len(refPts) > ptLength:
            image = cv2.circle(image, refPts[-1], radius=2, color=(0, 255, 0), thickness=-1)
            if len(refPts)>1:
                image = cv2.line(image,refPts[-2],refPts[-1], color=(0, 255, 0), thickness=1)
        
        if key == ord("q") or len(refPts) >nPts:
            refPts.pop(-1) # last click is just to exit
            cv2.destroyAllWindows()
            break
        
    # get image height and subtract y
    ht = image.shape[1]
    # refPts = [[rP[0],ht - rP[1]] for rP in refPts]
    
    return refPts


def sampleGrid(imagePoints, gridSize = (5,3),imagePath = None,gridDims = [(-40,40),(0,60),(0,0)]):
    
    if len(imagePoints) != 6:
        raise Exception('grid generation is hard coded to only take 6 points.')
    
    # create a grid of points between the points in imagePoints
    # nInterp is the number of points to interpolate between each pair of points
    # 3d points start at the origin and are a gridSize grid in the x-y plane.
    # the z coordinate is the same for all points

    # create a grid of points in 3d
    x = np.linspace(gridDims[0][0],gridDims[0][1],gridSize[0])
    y = np.linspace(gridDims[1][0],gridDims[1][1],gridSize[1])
    z = 0
    X,Y,Z = np.meshgrid(x,y,z)

    # reshape to a list of points
    points3d = np.vstack((X.flatten(),Y.flatten(),Z.flatten())).T

    # create a list of points2d as midpoint between imagePoints  
    imagePoints = np.array(imagePoints)
    
    vec0 = np.ndarray((gridSize[0],2))
    vec0[0,:] = imagePoints[0]
    vec0[1,:] = np.divide((imagePoints[0]+imagePoints[1]),2)
    vec0[2,:] = imagePoints[1]
    vec0[3,:] = np.divide((imagePoints[1]+imagePoints[2]),2)
    vec0[4,:] = imagePoints[2]
    
    vec2 = np.ndarray((gridSize[0],2))
    vec2[0,:] = imagePoints[3]
    vec2[1,:] = np.divide((imagePoints[3]+imagePoints[4]),2)
    vec2[2,:] = imagePoints[4]
    vec2[3,:] = np.divide((imagePoints[4]+imagePoints[5]),2)
    vec2[4,:] = imagePoints[5]

    # vec1 is the midpoint between vec0 and vec2
    vec1 = (vec0 + vec2)/2
    
    points2d = np.vstack((vec0,vec1,vec2))

    return points2d, points3d

def manualExtrinsicCalibration(videoPath,gridDims3d,cameraParametersPath,
                               savePathList):

    # pop an image
    utilsChecker.video2Images(videoPath, nImages=2, tSingleImage=.1, 
                              filePrefix='output', outputFolder='default')
    
    imagePath = os.path.join(os.path.dirname(videoPath),'output0.png')
    
    picked2dPoints = pickPointsFromImage(imagePath,6)
    
    points2d, points3d = sampleGrid(picked2dPoints, gridDims = gridDims3d)
    
    # load in some camera params for intrinsics
    CameraParams = utils.loadCameraParameters(cameraParametersPath)
    
    # pass to calibration function
    manualCameraParams = utilsChecker.calcExtrinsics(imagePath, CameraParams, None,
                                                   points2d=points2d,points3d=points3d)
    
    # save these custom extrinsics to both camera folder and Calibration folder
    
    for savePath in savePathList:
        utilsChecker.saveCameraParameters(savePath,manualCameraParams)
        print('Camera parameters saved to ' + savePath)
        
    return manualCameraParams
# %% 
def getTrialNames(path):
    # get all the folder names in the path
    return [f.name for f in os.scandir(path) if f.is_dir()]
