"""
---------------------------------------------------------------------------
OpenCap: processViconVideos.py
---------------------------------------------------------------------------

Copyright 2023 Stanford University and the Authors

Author(s): Scott Uhlrich, Antoine Falisse

Licensed under the Apache License, Version 2.0 (the "License"); you may not
use this file except in compliance with the License. You may obtain a copy
of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This script processes videos from Vicon RGB cameras to estimate kinematics.
"""

# %% Paths and imports.
import os
import sys
import shutil
import utilsMocap

repoDir = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),'../'))
sys.path.append(repoDir)

from main import main


# %% User inputs
# Enter the path to the folder where you downloaded the data. The data is on
# SimTK: https://simtk.org/frs/?group_id=2385 (LabValidation_withVideos).
# In this example, our path looks like:
#   C:/Users/opencap/Documents/LabValidation_withVideos/subject2
#   C:/Users/opencap/Documents/LabValidation_withVideos/subject3
#   ...
dataBaseDir = 'C:\SharedGdrive/HPL_MASPL/'

# rewrite video directories?
overwriteRestructuring = True
deleteFolders = False
copyVideos = False # true to copy into new structure
saveCameraVolumeAnimation = False


# The dataset includes 2 sessions per subject.The first session includes
# static, sit-to-stand, squat, and drop jump trials. The second session 
# includes walking trials. The sessions are named <subject_name>_Session0 and 
# <subject_name>_Session1.
sessionNames = ['1003']

# TODO -> do this systematically
height = 1.75
mass = 65.8 
subjectID = '1003'

# We only support OpenPose on Windows.
poseDetector = 'OpenPose'

# List of cameras in order. First entry will be Cam 0.
camList = ['2122194','2111275','2141511','2111270','2121724']

# Select the camera configuration you would like to use.
# cameraSetups = ['2-cameras', '3-cameras', '5-cameras']
cameraSetups = ['2-cameras']

# Select the resolution at which you would like to use OpenPose. More details
# about the options in Examples/reprocessSessions. In the paper, we compared 
# 'default' and '1x1008_4scales'.
resolutionPoseDetection = '1x1008_4scales'

# Since the prepint release, we updated a new augmenter model. To use the model
# used for generating the paper results, select v0.1. To use the latest model
# (now in production), select v0.2.
augmenter_model = 'v0.2'

# %% Data re-organization
# To reprocess the data, we need to re-organize the data so that the folder
# structure is the same one as the one expected by OpenCap. It is only done
# once as long as the variable overwriteRestructuring is False. 
# We also convert the Vicon calibration to OpenCap format and put in the 
# appropriate OpenCap folder.
subjects = ['S' + sN for sN in sessionNames]
for subject in subjects:
    pathRawVideos = os.path.join(dataBaseDir,'RawData',subject,'Videos')
    outPath = os.path.join(dataBaseDir,'OpenCap','Data',subject)
    if not os.path.exists(os.path.join(outPath,'Videos','Cam0')) or overwriteRestructuring:
        if overwriteRestructuring and deleteFolders:
            try:
                shutil.rmtree(os.path.join(outPath,'Videos'))
                print('deleted output file structure')
            except:
                pass
        utilsMocap.moveFilesToOpenCapStructure(pathRawVideos, outPath, 
                                               camList, calibration=True,
                                               saveAnimation=saveCameraVolumeAnimation,
                                               copyVideos=copyVideos)

    # Save metadata if not already done
    utilsMocap.createMetadata(outPath, height=height, mass=mass, subjectID=subjectID)

# %% Fixed settings.
# The dataset contains 5 videos per trial. The 5 videos are taken from cameras
# positioned at different angles: Cam0:-70deg, Cam1:-45deg, Cam2:0deg, 
# Cam3:45deg, and Cam4:70deg where 0deg faces the participant. Depending on the
# cameraSetup, we load different videos.
cam2sUse = {'5-cameras': ['Cam0', 'Cam1', 'Cam2', 'Cam3', 'Cam4'], 
            '3-cameras': ['Cam1', 'Cam2', 'Cam3'], 
            '2-cameras': ['Cam1', 'Cam3']}

# # %% Functions for re-processing the data.
def process_trial(trial_name=None, session_name=None, isDocker=False,
                  cam2Use=['all'],
                  intrinsicsFinalFolder='Deployed', extrinsicsTrial=False,
                  alternateExtrinsics=None, markerDataFolderNameSuffix=None,
                  imageUpsampleFactor=4, poseDetector='OpenPose',
                  resolutionPoseDetection='default', scaleModel=False,
                  bbox_thr=0.8, augmenter_model='v0.2', benchmark=False,
                  calibrationOptions=None, offset=True, dataDir=None):

    # Run main processing pipeline.
    main(session_name, trial_name, trial_name, cam2Use, intrinsicsFinalFolder,
          isDocker, extrinsicsTrial, alternateExtrinsics, calibrationOptions,
          markerDataFolderNameSuffix, imageUpsampleFactor, poseDetector,
          resolutionPoseDetection=resolutionPoseDetection,
          scaleModel=scaleModel, bbox_thr=bbox_thr,
          augmenter_model=augmenter_model, benchmark=benchmark, offset=offset,
          dataDir=dataDir)

    return

# Hard-coded way through for now

sessionName = 'S1003'
cam2Use = ['Cam' + str(i) for i in [0, 1, 4]]
markerDataFolderNameSuffix = '3-cameras-014'
dataDir = 'C:/SharedGdrive/HPL_MASPL/OpenCap'



# get all
# trialNames = utilsMocap.getTrialNames(os.path.join(dataDir, 'Data', sessionName, 'Videos', 'Cam0', 'InputMedia'))

# specify trialNames
trialNames = ['SLSQUATS_R','SLSQUATS_L']
trialNames = []
trialNames = [sessionName + '_' + t for t in trialNames]


# for Scaling
scaleModel = [True]
staticTrial = [sessionName + '_DLSQUATS']
# scaleModel = []
# staticTrial = []
trials = staticTrial + trialNames
scaleModel = scaleModel + [False for i in range(len(trialNames))]


for i,trial in enumerate(trials):
    process_trial(trial,
                    session_name=sessionName,
                    cam2Use=cam2Use, 
                    extrinsicsTrial=False,
                    markerDataFolderNameSuffix=markerDataFolderNameSuffix,
                    poseDetector=poseDetector,
                    resolutionPoseDetection=resolutionPoseDetection,
                    scaleModel=scaleModel[i], 
                    augmenter_model=augmenter_model,
                    dataDir=dataDir)

test =1























# %% Process trials.
# for count, sessionName in enumerate(sessionNames):    
#     # Get trial names.
#     pathCam0 = os.path.join(dataDir, 'Data', sessionName, 'Videos', 'Cam0',
#                             'InputMedia')    
#     # Work around to re-order trials and have the extrinsics trial firs, and
#     # the static second (if available).
#     trials_tmp = os.listdir(pathCam0)
#     trials_tmp = [t for t in trials_tmp if
#                   os.path.isdir(os.path.join(pathCam0, t))]
#     session_with_static = False
#     for trial in trials_tmp:
#         if 'extrinsics' in trial.lower():                    
#             extrinsics_idx = trials_tmp.index(trial) 
#         if 'static' in trial.lower():                    
#             static_idx = trials_tmp.index(trial) 
#             session_with_static = True            
#     trials = [trials_tmp[extrinsics_idx]]
#     if session_with_static:
#         trials.append(trials_tmp[static_idx])
#         for trial in trials_tmp:
#             if ('static' not in trial.lower() and 
#                 'extrinsics' not in trial.lower()):
#                 trials.append(trial)
#     else:
#         for trial in trials_tmp:
#             if 'extrinsics' not in trial.lower():
#                 trials.append(trial)
    
#     for poseDetector in poseDetectors:
#         for cameraSetup in cameraSetups:
#             cam2Use = cam2sUse[cameraSetup]
            
#             # The second sessions (<>_1) have no static trial for scaling the
#             # model. The static trials were collected as part of the first
#             # session for each subject (<>_0). We here copy the Model folder
#             # from the first session to the second session.
#             if sessionName[-1] == '1':
#                 sessionDir = os.path.join(dataDir, 'Data', sessionName)
#                 sessionDir_0 = sessionDir[:-1] + '0'
#                 camDir_0 = os.path.join(
#                     sessionDir_0, 'OpenSimData', 
#                     poseDetector + '_' + resolutionPoseDetection, cameraSetup)
#                 modelDir_0 = os.path.join(camDir_0, 'Model')
#                 camDir_1 = os.path.join(
#                     sessionDir, 'OpenSimData', 
#                     poseDetector + '_' + resolutionPoseDetection, cameraSetup)
#                 modelDir_1 = os.path.join(camDir_1, 'Model')
#                 os.makedirs(modelDir_1, exist_ok=True)
#                 for file in os.listdir(modelDir_0):
#                     pathFile = os.path.join(modelDir_0, file)
#                     pathFileEnd = os.path.join(modelDir_1, file)
#                     shutil.copy2(pathFile, pathFileEnd)
                    
#             # Process trial.
#             for trial in trials:                
#                 print('Processing {}'.format(trial))
                
#                 # Detect if extrinsics trial to compute extrinsic parameters. 
#                 if 'extrinsics' in trial.lower():                    
#                     extrinsicsTrial = True
#                 else:
#                     extrinsicsTrial = False
                
#                 # Detect if static trial with neutral pose to scale model.
#                 if 'static' in trial.lower():                    
#                     scaleModel = True
#                 else:
#                     scaleModel = False
                                   
                    
#                 process_trial(trial,
#                               session_name=sessionName,
#                               cam2Use=cam2Use, 
#                               extrinsicsTrial=False,
#                               markerDataFolderNameSuffix=cameraSetup,
#                               poseDetector=poseDetector,
#                               resolutionPoseDetection=resolutionPoseDetection,
#                               scaleModel=scaleModel, 
#                               augmenter_model=augmenter_model,
#                               dataDir=dataDir)