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

repoDir = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),'../'))
sys.path.append(repoDir)

from labValidationIDs import getData
from utils import downloadVideosFromServer, getDataDirectory
from main import main

# %% User inputs

# The dataset includes 2 sessions per subject.The first session includes
# static, sit-to-stand, squat, and drop jump trials. The second session 
# includes walking trials. The sessions are named <subject_name>_0 and 
# <subject_name>_1.
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
cameraSetups = ['2-cameras', '3-cameras', '5-cameras']

# Select the resolution at which you would like to use OpenPose. More details
# about the options in Examples/reprocessSessions. In the paper, we compared 
# 'default' and '1x1008_4scales'.
resolutionPoseDetection = 'default'

# %% Functions for re-processing the data.
def process_trial(trial_id, trial_name=None, session_name='', isDocker=False,
                  session_id=None, cam2Use=['all'],
                  intrinsicsFinalFolder='Deployed', extrinsicsTrial=False,
                  alternateExtrinsics=None, markerDataFolderNameSuffix=None,
                  imageUpsampleFactor=4, poseDetector='OpenPose',
                  resolutionPoseDetection='default', scaleModel=False,
                  bbox_thr=0.8, augmenter_model='v0.2', benchmark=False,
                  calibrationOptions=None):
    
    # Download videos.
    trial_name = downloadVideosFromServer(session_id, trial_id, isDocker=True, 
                                          trial_name=trial_name,
                                          session_name=session_name,
                                          benchmark=benchmark)

    # Run main processing pipeline.
    main(session_name, trial_name, trial_id, cam2Use, intrinsicsFinalFolder,
         isDocker, extrinsicsTrial, alternateExtrinsics, calibrationOptions,
         markerDataFolderNameSuffix, imageUpsampleFactor, poseDetector,
         resolutionPoseDetection=resolutionPoseDetection,
         scaleModel=scaleModel, bbox_thr=bbox_thr,
         augmenter_model=augmenter_model, benchmark=benchmark)

    return

# %% Process trials.
for count, sessionName in enumerate(sessionNames):
    for poseDetector in poseDetectors:
        for cameraSetup in cameraSetups:
            data = getData(sessionName)
            cam2Use = data['camera_setup'][cameraSetup]

            # The second sessions (<>_1) have no static trial for scaling the
            # model. The static trials were collected as part of the first
            # session for each subject (<>_0). We here copy the Model folder
            # from the first session to the second session.
            if sessionName[-1] == '1':
                dataDir = getDataDirectory()
                sessionDir = os.path.join(dataDir, 'Data', sessionName)
                sessionDir_0 = sessionDir[:-1] + '0'
                camDir_0 = os.path.join(
                    sessionDir_0, 'OpenSimData', 
                    poseDetector + '_' + resolutionPoseDetection, cameraSetup)
                modelName = 'LaiArnoldModified2017_poly_withArms_weldHand'
                modelDir_0 = os.path.join(camDir_0, modelName, 'Model')
                camDir_1 = os.path.join(
                    sessionDir, 'OpenSimData', 
                    poseDetector + '_' + resolutionPoseDetection, cameraSetup)
                modelDir_1 = os.path.join(camDir_1, modelName, 'Model')
                os.makedirs(modelDir_1, exist_ok=True)
                for file in os.listdir(modelDir_0):
                    pathFile = os.path.join(modelDir_0, file)
                    pathFileEnd = os.path.join(modelDir_1, file)
                    shutil.copy2(pathFile, pathFileEnd)
                
            # Process trial.
            for trial in data['trials']:
                
                name = None # default
                if "name" in data['trials'][trial]:
                    name = data['trials'][trial]["name"]
                    
                intrinsicsFinalFolder = 'Deployed' # default
                if "intrinsicsFinalFolder" in data['trials'][trial]:
                    intrinsicsFinalFolder = (
                        data['trials'][trial]["intrinsicsFinalFolder"])
                    
                extrinsicsTrial = False # default
                if "extrinsicsTrial" in data['trials'][trial]:
                    extrinsicsTrial = data['trials'][trial]["extrinsicsTrial"]
                    
                scaleModel = False # default
                if "scaleModel" in data['trials'][trial]:
                    scaleModel = data['trials'][trial]['scaleModel']
                    
                # Select 'v0.1' to use the augmenter model used for the paper
                # results. We re-trained the model, and obtained better results
                # as compared to the paper results. We therefore now default to
                # the re-trained model.
                augmenter_model = 'v0.2' # default
                if "augmenter_model" in data['trials'][trial]:
                    augmenter_model = data['trials'][trial]['augmenter_model']
                    
                process_trial(data['trials'][trial]["id"], name,
                              session_name=sessionName,
                              session_id=data['session_id'],
                              isDocker=False, cam2Use=cam2Use, 
                              intrinsicsFinalFolder=intrinsicsFinalFolder,
                              extrinsicsTrial=extrinsicsTrial,
                              markerDataFolderNameSuffix=cameraSetup,
                              poseDetector=poseDetector,
                              resolutionPoseDetection=resolutionPoseDetection,
                              scaleModel=scaleModel, 
                              augmenter_model=augmenter_model)