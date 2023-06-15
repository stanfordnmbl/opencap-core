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
resolution pose estimation settings It is also useful for debugging failed trials.


The example session is publicly available, but you will need login credentials
from app.opencap.ai to run this script. You can quickly create credentials
from the home page: https://app.opencap.ai. We recommend first running the 
Examples/createAuthenticationEnvFile.py script prior to running this script. Otherwise,
you will need to to login every time you run the script.


You can view the example session at:
https://app.opencap.ai/session/23d52d41-69fe-47cf-8b60-838e4268dd50
"""

import os
import sys
sys.path.append(os.path.abspath('./..'))

from utilsServer import batchReprocess
from utils import getTrialNameIdMapping
from utilsAPI import getAPIURL
from utilsAuth import getToken

API_URL = getAPIURL()
API_TOKEN = getToken()

# %% User inputs.
# Enter the identifier(s) of the session(s) you want to reprocess. This is a list of one
# or more session identifiers. The identifier is found as the 36-character string at the
# end of the session url: app.opencap.ai/session/<session_id>
session_ids = ['d9ce6b4c-be0b-4685-85a6-66e2776bdb7f']



# Select which trials to reprocess. You can reprocess all trials in the session 
# by entering None in all fields below. The correct calibration and static
# trials will be automatically selected if None, and all dynamic trials will be
# processed if None. If you do not want to process one of the trial types, 
# enter []. If you specify multiple sessions above, all of the fields
# below must be None or []. If you selected only one session_id above, you may
# select specific reprocessed. Only one trial (str) is allowed for calib_id and
# static_id. A list of strings is allowed for dynamic_ids.

#calibration is the calibration of the box
calib_id = [] # None (auto-selected trial), [] (skip), or string of specific trial_id
#static is the scaling of the model in static pose
static_id = None # None (auto-selected trial), [] (skip), or string of specific trial_id
#dynamic is all other trials
dynamic_trialNames = ['upright_twoF0_trial0_1'] # None (all dynamic trials), [] (skip), or list of trial names

# extract trial ids from trial names
if dynamic_trialNames is not None and len(dynamic_trialNames)>0:
    trialNames = getTrialNameIdMapping(session_ids[0])
    dynamic_ids = [trialNames[name]['id'] for name in dynamic_trialNames]
else:
    dynamic_ids = dynamic_trialNames

# # Optional: Uncomment this section to create a list of dynamic_ids to reprocess.
# dynamic_ids = None # None (all dynamic trials), [] (skip), or list of trial_id strings

# The resolution at which the videos are processed by OpenPose can be adjusted.
# The finer the resolution the more accurate the results (typically) but also
# the more GPU memory is required and the more time it takes to process the video.
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
resolutionPoseDetection = 'default'


# Set deleteLocalFolder to False to keep a local copy of the data. If you are 
# reprocessing a session that you collected, data will get written to the database
# regardless of your selection. If True, the local copy will be deleted.
deleteLocalFolder = False
      

# %% Process data.
batchReprocess(session_ids,calib_id,static_id,dynamic_ids,
               resolutionPoseDetection=resolutionPoseDetection,
               deleteLocalFolder=deleteLocalFolder)
