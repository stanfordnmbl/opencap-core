"""
---------------------------------------------------------------------------
OpenCap: makeSessionsPublic.py
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


This script makes sessions public or not public. You must be logged in as the 
session owner to perform this operation.
"""

import os
import sys

sys.path.append(os.path.abspath('./..'))

from utilsAuth import getToken
from utilsAPI import getAPIURL
from utils import makeSessionPublic

API_URL = getAPIURL()
API_TOKEN = getToken()

sessionList = ['<yourSessionID1>','<yourSessionID2>'] # input list of session identifier strings
makePublic = True # True to make it public, False to make it private
       
for session in sessionList:
    
    makeSessionPublic(session,publicStatus=makePublic) 