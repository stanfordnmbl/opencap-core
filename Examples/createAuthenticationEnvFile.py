'''
    ---------------------------------------------------------------------------
    OpenCap: createAuthenticationEnvFile.py
    ---------------------------------------------------------------------------

    Copyright 2022 Stanford University and the Authors
    
    Author(s): Antoine Falisse, Scott Uhlrich
    
    Licensed under the Apache License, Version 2.0 (the "License"); you may not
    use this file except in compliance with the License. You may obtain a copy
    of the License at http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    
    
    Run this script to save your authentication token (for logging into webapp)
    as an environment variable in a .env file. 
    DO NOT CHANGE THE FILE NAME OF THE .env FILE, or your credentials will get pushed
    to github with a commit, and anyone could get access to your data.
    We recommend only saving this .env file on your own encrypted machine
    and not while running on google collab.
antoinefalisse'''

import os
import sys
sys.path.append(os.path.abspath('./..'))

from utilsAuth import getToken

# Restart your terminal or IDE (eg Spyder) after running the script for the 
# new environment variable to be loaded. 
getToken(saveEnvPath=os.path.abspath(os.path.join(os.path.dirname(__file__),'../')))