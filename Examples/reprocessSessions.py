"""
---------------------------------------------------------------------------
OpenCap: reprocessSessions.py
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


This script reprocesses OpenCap trials collected with the web application at
app.opencap.ai. You can specify multiple sessions to reprocess, or specific
trials within a session to reprocess. Data will (optionally) be saved locally
in your <repositoryDirectory>/Data/<session_id> folder. 


This script is useful for processing a session with more accurate and higher 
resolution pose estimation settings. It is also useful for debugging failed
trials.


The example session is publicly available, but you will need login credentials
from app.opencap.ai to run this script. You can quickly create credentials
from the home page: https://app.opencap.ai. We recommend first running the 
Examples/createAuthenticationEnvFile.py script prior to running this script.
Otherwise, you will need to to login every time you run the script.


You can view the example session at:
https://app.opencap.ai/session/23d52d41-69fe-47cf-8b60-838e4268dd50
"""

import os
import sys
sys.path.append(os.path.abspath('./..'))

from utilsServer import batchReprocess
from utilsAPI import getAPIURL
from utilsAuth import getToken

API_URL = getAPIURL()
API_TOKEN = getToken()

# %% User inputs.
# Enter the identifier(s) of the session(s) you want to reprocess. This is a list of one
# or more session identifiers. The identifier is found as the 36-character string at the
# end of the session url: app.opencap.ai/session/<session_id>
session_ids = ['9eea5bf0-a550-4fa5-bc69-f5f072765848']

# Select which trials to reprocess. You can reprocess all trials in the session 
# by entering None in all fields below. The correct calibration and static
# trials will be automatically selected if None, and all dynamic trials will be
# processed if None. If you do not want to process one of the trial types, 
# enter []. If you specify multiple sessions above, all of the fields
# below must be None or []. If you selected only one session_id above, you may
# select specific trials. Only one trial (str) is allowed for calib_id and
# static_id. A list of strings is allowed for dynamic_trialNames.

calib_id = [] # None (auto-selected trial), [] (skip), or string of specific trial_id
static_id = [] # None (auto-selected trial), [] (skip), or string of specific trial_id
dynamic_trialNames = ['jump','jump2'] # None (all dynamic trials), [] (skip), or list of trial names

# Select which pose estimation model to use; options are 'OpenPose' and 'hrnet'.
# If the same pose estimation model was used when collecting data with the web
# app and if (OpenPose only) you are not reprocessing the data with a different
# resolution, then the already computed pose estimation results will be used and
# pose estimation will not be re-run. Please note that we do not provide support
# for running 'hrnet' locally. If you want to use 'hrnet', you will need to have
# selected 'hrnet' when collecting data with the web app. You can however re-
# process data originally collected with 'hrnet' with 'OpenPose' if you have 
# installed OpenPose locally (see README.md for instructions).
poseDetector = 'hrnet'

# OpenPose only:
# Select the resolution at which the videos are processed. There are no
# resolution options for hrnet and resolutionPoseDetection will be ignored.
# The finer the resolution the more accurate the results (typically) but also
# the more GPU memory is required and the more time it takes for processing.
# OpenCap supports the following four resolutionPoseDetection options (ordered
# from lower to higher resolution/required memory):
#   - 'default': 1x368 resolution, default in OpenPose (we were able to run with a GPU with 4GB memory).
#   - '1x736': 1x736 resolution, default in OpenCap (we were able to run with a GPU with 4GB memory).
#   - '1x736_2scales': 1x736 resolution with 2 scales (gap = 0.75). (may help with people larger in the frame. (we were able to run with a GPU with 8GB memory)
#       - Please visit https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/include/openpose/flags.hpp#L112
#         to learn more about scales. The conversation in this issue is also
#         relevant: https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/622
#   - '1x1008_4scales': 1x1008 resolution with 4 scales (gap = 0.25). (we were only able to run with a GPU with 24GB memory)
#       - This is the highest resolution/settings we could use with a 24GB
#         GPU without running into memory issues.
resolutionPoseDetection = '1x736'


# Set deleteLocalFolder to False to keep a local copy of the data. If you are 
# reprocessing a session that you collected, data will get written to the database
# regardless of your selection. If True, the local copy will be deleted.
deleteLocalFolder = False
      

# %% Process data.

reprocessDict = {
    # 'a118e69e-f652-42e0-9b4c-009ba60b4f86':['10mwt'], 
    # '2e0b9298-4036-4fe6-ad88-cdbccc4cca57':['brooke','10mwt'],
    # '952ea0bd-61ba-428c-9dbe-491fe37329f0':['10mwt'], 
    # 'a4679c46-2f38-4cc5-86b4-bdd3f01791ab':['tug_cone'], 
    # # '1fc22b0c-1ef8-4b6d-b83b-85b12634905d':['10mwrt_1'], # didn't help
    # 'c70f88f1-c189-40f9-8fa5-41d673a8708d':['10mwrt'], 
    # '8b7b2366-0d1e-4336-a867-59d9b6e576de':['10mwrt'],
    # '109eabdf-e50d-459e-8917-c856338a720b':['10mwt','10mwrt'],
    # '9c4666be-57f4-4a6b-9c51-7e394f4a53e1':['10mwt'],
    # 'a7675b05-88da-4bb0-b27b-5357c3ec5807':['10mwrt'], 
    # 'ab14edea-6d5d-4904-af7d-25f392f296b0':['brooke','tug_cone','10mwrt'], 
    # '04cad9b0-796a-40ec-aa24-346c71e8ff37':['10mwt'],
    # '84aef447-6982-4842-8f95-e63459993fcf':['10mwt'],
    
    # # MDF

    # 'c2001341-2624-4595-91d1-9af78ed76421':['10mwt_1'],
    # 'f5ec71e8-7898-4c51-993b-897014a3e8e3':['10mwrt'],
    # 'ee86be93-b1da-4080-b445-176f6071e734':['10mwt_2','10mwrt'],
    # '8963b080-d316-4183-8928-47bebcac70b1':['10mwrt'],
    # '00faec9c-71e8-4051-8175-612c2488b0bb':['5xsts_1'], # no people in background. should be fine; may just be ood
    # 'cda2db6e-b268-42ee-99d0-9cc358e893d1':['10mwrt'],
    # '30ef567b-6485-447c-a5c8-b99ed84c17a4':['10mwt'],
    # '6c071107-3735-4c71-a03a-68cac9aa0546':['brooke'],
    # '551a60a5-11cb-49a5-879b-477cf499af7a':['10mrt'],
    # 'a384272c-bc90-4150-ab94-a2af8f5a9315':['10mwrt','10mwt_2'],
    'd7793a0f-1451-4e21-a2d2-3d3f1bd9c7de':['5xsts'], # START REVIEWING
    '43db734e-13fb-4d6d-9223-09f967dd40f0':['curls','toe_stand'],
    'b9578e78-d717-49de-a226-7f797dcc43a3':['10mwrt'],
    '5e16a747-e5ca-4853-98f6-a449452c494d':['brooke','10mrt'],
    'ace248c9-61a6-4c41-93eb-83edd823de0c':['toe_stand'],
    '76b201f8-1950-414e-a48a-97bf932c61bc':['10mwt'],
    '90e3af8d-3aac-4d97-ac5a-70f5c4e911ae':['brooke','10mwt','5xsts'],
    '4769f868-d487-4e0d-bda6-98eb6751b40a':['toe_stand','jump'],
    '91ef0085-13a1-49fe-8bcf-91e37efc4d53':['tug_cone'],
    '9e22db8f-6356-46c0-a118-d8f2741f97be':['10mwt','10mrt','tug_cone'],
    '57e48655-deab-447a-9d0c-c292b124fdbd':['10mwt_1'],
    '34f2095f-d076-4a3d-b094-75f968a93c21':['jump'],
    'c0bf6608-e35d-40a8-9ca2-2abc4a9c3590':['cup_weight','10mwt','10mrt','tug_cone','5xsts'],
    '9808e3d7-b2a1-4864-a3c7-7e6549a5dd36':['10mwt'],
    '513ec455-04ba-4bc4-b502-5f91d34c4ce6':['brooke','10mrt','tug_cone','5xsts'],
    '0207a325-f5cb-4b26-8544-a93d9831552b':['10mwt'],
    '6ff20ae9-7f99-4837-95e7-82cedb242522':['10mrt','tug_cone'],
    'f9a5c172-288a-4886-b3bb-417ab3f48b55':['10mwt'],
    '78d9fbfe-04e0-4766-ba15-198e246d5e9c':['brooke','10mwt','10mrt'],
    '41599e7a-8eea-4b03-b5ef-57502504e879':['toe_stand'],
    '9871c398-0d02-450b-86a6-0d8c6b27b26d':['10mwt','5xsts'],
    '621e5ae8-226b-4fc6-bbd9-df448328ff1f':['jump_1'],
    'aca92056-3f67-403c-8d6a-513055274ffe':['brooke','tug_cone','5xsts'],
    '25d785ae-f49e-4531-b14c-86d6d5e9d144':['toe_stand']
    }

# for session_id, dynamic_trialNames in reprocessDict.items():
    
#     session_ids = [session_id]
#     batchReprocess(session_ids,calib_id,static_id,dynamic_trialNames,
#                     poseDetector=poseDetector,
#                     resolutionPoseDetection=resolutionPoseDetection,
#                     deleteLocalFolder=deleteLocalFolder)

batchReprocess(session_ids,calib_id,static_id,dynamic_trialNames,
                    poseDetector=poseDetector,
                    resolutionPoseDetection=resolutionPoseDetection,
                    deleteLocalFolder=deleteLocalFolder)

test = 1