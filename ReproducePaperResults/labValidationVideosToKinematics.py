"""
---------------------------------------------------------------------------
OpenCap: labValidationVideosToKinematics.py
---------------------------------------------------------------------------

Copyright 2022 Stanford University and the Authors

Author(s): Scott Uhlrich, Antoine Falisse

Licensed under the Apache License, Version 2.0 (the "License"); you may not
use this file except in compliance with the License. You may obtain a copy
of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This script processes videos to estimate kinematics with the data from the 
"OpenCap: 3D human movement dynamics from smartphone videos" paper. The dataset
includes 10 subjects, with 2 sessions per subject. The first session includes
static, sit-to-stand, squat, and drop jump trials. The second session includes
walking trials. The sessions are named <subject_name>_0 and <subject_name>_1.
The data for this validation part of the paper were collected prior to 
developing the web application. The session and trial IDs were therefore
manually entered, see labValidationIDs. In the paper, we compared 3 camera
configurations: 2 cameras at +/- 45deg ('2-cameras'), 3 cameras at +/- 45deg 
and 0deg ('3-cameras'), and 5 cameras at +/- 45deg, +/- 70deg, and 0deg
('5-cameras'); 0deg faces the participant. Use the variable cameraSetups below
to select which camera configuration to use. In the paper, we also compared
three algorithms for pose detection: OpenPose at default resolution, OpenPose
with higher accuracy, and HRNet. HRNet is not supported on Windows, and we
therefore do not support it here (it is supported on the web application). To
use OpenPose at default resolution, set the variable resolutionPoseDetection to
'default'. To use OpenPose with higher accuracy, set the variable 
resolutionPoseDetection to '1x1008_4scales'. Take a look at 
Examples/reprocessSessions for more details about OpenPose settings and the
GPU requirements. Please note that we have updated OpenCap since submitting
the paper. As part of the updates, we re-trained the deep learning model we
use to predict anatomical markers from video keypoints, and we updated how
videos are time synchronized. These changes might have a slight effect
on the results. By default, running this script will save the data in the
Data folder of your local repository.
"""

# %% Paths and imports.
import os
import sys
import shutil
import yaml

repoDir = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),'../'))
sys.path.append(repoDir)

from main import main
from utils import importMetadata

# %% User inputs
# Enter the path to the folder where you downloaded the data. The data is on
# SimTK: https://simtk.org/frs/?group_id=2385 (LabValidation_withVideos).
# In this example, our path looks like:
#   C:/Users/antoi/Documents/MyRepositories/mobilecap_data/Data/LabValidation/subject2
#   C:/Users/antoi/Documents/MyRepositories/mobilecap_data/Data/LabValidation/subject3
# ...
dataDir = 'C:/Users/antoi/Documents/MyRepositories/mobilecap_data/Data/LabValidation/'

# The dataset includes 2 sessions per subject.The first session includes
# static, sit-to-stand, squat, and drop jump trials. The second session 
# includes walking trials. The sessions are named <subject_name>_Session0 and 
# <subject_name>_Session1.
sessionNames = ['subject2_0', 'subject2_1',
                'subject3_0', 'subject3_1',
                'subject4_0', 'subject4_1',
                'subject5_0', 'subject5_1', 
                'subject6_0', 'subject6_1',
                'subject7_0', 'subject7_1', 
                'subject8_0', 'subject8_1', 
                'subject9_0', 'subject9_1', 
                'subject10_0', 'subject10_1', 
                'subject11_0', 'subject11_1']

# We only support OpenPose on Windows.
poseDetectors = ['OpenPose']

# Select the camera configuration you would like to use.
# cameraSetups = ['2-cameras', '3-cameras', '5-cameras']
cameraSetups = ['2-cameras']

# Select the resolution at which you would like to use OpenPose. More details
# about the options in Examples/reprocessSessions. In the paper, we compared 
# 'default' and '1x1008_4scales'.
resolutionPoseDetection = 'default'

# Since the prepint release, we updated a new augmenter model. To use the model
# used for generating the paper results, select v0.1. To use the latest model
# (now in production), select v0.2.
augmenter_model = 'v0.1'

# %% Data re-organization
# To reprocess the data, we need to re-organize the data so that the folder
# structure is the same one as the one expected by OpenCap. It is only done
# once as long as the variable overwriteRestructuring is False. To overwrite
# filpt the flag to True.
overwriteRestructuring = False
subjects = ['subject' + str(i) for i in range(2,12)]
for subject in subjects:
    pathSubject = os.path.join(dataDir, subject)
    pathVideos = os.path.join(pathSubject, 'VideoData')    
    for session in os.listdir(pathVideos):
        if 'Session' not in session:
            continue
        pathSession = os.path.join(pathVideos, session)
        pathSessionNew = os.path.join(dataDir, 'Data', subject + '_' + session)
        if os.path.exists(pathSessionNew) and not overwriteRestructuring:
            continue
        os.makedirs(pathSessionNew, exist_ok=True)
        # Copy metadata
        pathMetadata = os.path.join(pathSubject, 'metadata.yaml')
        shutil.copy2(pathMetadata, pathSessionNew)
        pathMetadataNew = os.path.join(pathSessionNew, 'metadata.yaml')
        pathMetadataNewRenamed = os.path.join(pathSessionNew, 
                                              'sessionMetadata.yaml')
        os.rename(pathMetadataNew, pathMetadataNewRenamed)
        # Adjust model name
        sessionMetadata = importMetadata(pathMetadataNewRenamed)
        sessionMetadata['openSimModel'] = (
            'LaiArnoldModified2017_poly_withArms_weldHand')
        with open(pathMetadataNewRenamed, 'w') as file:
                yaml.dump(sessionMetadata, file)        
        for cam in os.listdir(pathSession):
            if "Cam" not in cam:
                continue            
            pathCam = os.path.join(pathSession, cam)
            pathCamNew = os.path.join(pathSessionNew, 'Videos', cam)
            pathInputMediaNew = os.path.join(pathCamNew, 'InputMedia')
            # Copy videos.
            for trial in os.listdir(pathCam):
                pathTrial = os.path.join(pathCam, trial)
                if not os.path.isdir(pathTrial):
                    continue
                pathVideo = os.path.join(pathTrial, trial + '.avi')
                pathTrialNew = os.path.join(pathInputMediaNew, trial)
                os.makedirs(pathTrialNew, exist_ok=True)
                shutil.copy2(pathVideo, pathTrialNew)
            # Copy camera parameters
            pathParameters = os.path.join(pathCam, 
                                          'cameraIntrinsicsExtrinsics.pickle')
            shutil.copy2(pathParameters, pathCamNew)

# %% Fixed settings.
# The dataset contains 5 videos per trial. The 5 videos are taken from cameras
# positioned at different angles: Cam0:-70deg, Cam1:-45deg, Cam2:0deg, 
# Cam3:45deg, and Cam4:70deg where 0deg faces the participant. Depending on the
# cameraSetup, we load different videos.
cam2sUse = {'5-cameras': ['Cam0', 'Cam1', 'Cam2', 'Cam3', 'Cam4'], 
            '3-cameras': ['Cam1', 'Cam2', 'Cam3'], 
            '2-cameras': ['Cam1', 'Cam3']}

# %% Functions for re-processing the data.
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

# %% Process trials.
for count, sessionName in enumerate(sessionNames):    
    # Get trial names.
    pathCam0 = os.path.join(dataDir, 'Data', sessionName, 'Videos', 'Cam0',
                            'InputMedia')    
    # Work around to re-order trials and have the static first (if available).
    trials_tmp = os.listdir(pathCam0)
    trials_tmp = [t for t in trials_tmp if
                  os.path.isdir(os.path.join(pathCam0, t))]
    # Re-order to have static first
    session_with_static = False
    for trial in trials_tmp:
        if 'static' in trial.lower():                    
            static_idx = trials_tmp.index(trial) 
            session_with_static = True
    if session_with_static:
        trials = [trials_tmp[static_idx]]
        for trial in trials_tmp:
            if 'static' not in trial.lower():
                trials.append(trial)
    else:
        trials = trials_tmp
    
    for poseDetector in poseDetectors:
        for cameraSetup in cameraSetups:
            cam2Use = cam2sUse[cameraSetup]

            # The second sessions (<>_1) have no static trial for scaling the
            # model. The static trials were collected as part of the first
            # session for each subject (<>_0). We here copy the Model folder
            # from the first session to the second session.
            if sessionName[-1] == '1':
                sessionDir = os.path.join(dataDir, 'Data', sessionName)
                sessionDir_0 = sessionDir[:-1] + '0'
                camDir_0 = os.path.join(
                    sessionDir_0, 'OpenSimData', 
                    poseDetector + '_' + resolutionPoseDetection, cameraSetup)
                modelDir_0 = os.path.join(camDir_0, 'Model')
                camDir_1 = os.path.join(
                    sessionDir, 'OpenSimData', 
                    poseDetector + '_' + resolutionPoseDetection, cameraSetup)
                modelDir_1 = os.path.join(camDir_1, 'Model')
                os.makedirs(modelDir_1, exist_ok=True)
                for file in os.listdir(modelDir_0):
                    pathFile = os.path.join(modelDir_0, file)
                    pathFileEnd = os.path.join(modelDir_1, file)
                    shutil.copy2(pathFile, pathFileEnd)
                    
            # Process trial.
            for trial in trials:                
                print('Processing {}'.format(trial))
                
                # Detect if static trial with netural pose to scale model.
                if 'static' in trial.lower():                    
                    scaleModel = True
                else:
                    scaleModel = False
                
                # Session specific intrinsic parameters
                if 'subject2' in sessionName or 'subject3' in sessionName:
                    intrinsicsFinalFolder = 'Deployed_720_240fps'
                else:
                    intrinsicsFinalFolder = 'Deployed_720_60fps'
                    
                    
                process_trial(trial,
                              session_name=sessionName,
                              cam2Use=cam2Use, 
                              intrinsicsFinalFolder=intrinsicsFinalFolder,
                              markerDataFolderNameSuffix=cameraSetup,
                              poseDetector=poseDetector,
                              resolutionPoseDetection=resolutionPoseDetection,
                              scaleModel=scaleModel, 
                              augmenter_model=augmenter_model,
                              dataDir=dataDir)