# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 14:32:04 2023

Downloads data, reprocesses scaling/IK with new model, runs ID.
For NMD data collection

@author: Scott Uhlrich
"""

import sys
import os
repoDir = os.path.abspath('./..')
sys.path.append(repoDir)

import utils
from main import main
from utilsOpenSim import runIDTool

# # # # User inputs
# sessions to process
sessions = ['057d10da-34c7-4fb7-a127-6040010dde0']
trial_names = ['arm_rom']

# data directory
dataDir = 'C:/Users/suhlr/Documents/MyRepositories/sandbox/DATA'

# new model name. If using old model, set to None
newModelName = 'LaiUhlrich2022_shoulder'

# # # # #


for session in sessions:
    print('Processing session ' + session + '.')
    # Download data
    sessionDir = utils.downloadAndZipSession(session,justDownload=True,
                                             data_dir=dataDir)
    opensimDir = os.path.join(sessionDir, 'OpenSim')

    # load metadata
    sessionMetadata = utils.importMetadata(os.path.join(sessionDir,
                                                'sessionMetadata.yaml'))

    if newModelName != None:
        # change metadata
        oldModel = sessionMetadata['openSimModel']
        sessionMetadata ['opensimModel']= newModelName
        # save metadata
        utils.exportMetadata(sessionMetadata, os.path.join(sessionDir, 
                                                'sessionMetadata.yaml'))
    
        
        # re-run main for neutral trial - save as new model
        pathScaledModel = os.path.join(opensimDir, 'Models',newModelName + '_scaled.osim')
        neutralTrialID = utils.getNeutralTrial(session)

        main(session, 'neutral', neutralTrialID, isDocker=False, extrinsicsTrial=False,
                 poseDetector=sessionMetadata['poseDetector'],
                 scaleModel = True,
                 genericFolderNames = True)

        for trial_name in trial_names:
            trial_id = utils.getTrialId(session, trial_name)
            # re-run main for dynamic trials
            main(session, trial_name, trial_id, isDocker=False, extrinsicsTrial=False,
                    poseDetector=sessionMetadata['poseDetector'],
                    genericFolderNames = True)

        # change metadata back
        sessionMetadata ['opensimModel']= oldModel
        utils.exportMetadata(sessionMetadata, os.path.join(sessionDir, 
                                                'sessionMetadata.yaml'))
        
    else:
        pathScaledModel = os.path.join(opensimDir, 'Models', sessionMetadata['opensimModel'] + '_scaled.osim')

    for trial in trial_names:
        print('Processing trial ' + trial + '.')
        # run ID
        pathGenericSetupFileID = os.path.join(repoDir,'opensimPipeline','ID', 'SetupID.xml')
        pathScaledModel = os.path.join(opensimDir,'Models', sessionMetadata['opensimModel'] + '_scaled.osim')
        pathIKFile = os.path.join(opensimDir, 'IK', trial + '.mot')
        pathOutputFolder = os.path.join(opensimDir, 'Dynamics')
        os.makedirs(pathOutputFolder, exist_ok=True)

        # we don't have a GRF, EL, or time range, so we set it to None
        ELFile = None
        GRFFile = None
        timeRange = None

        print('Running Inverse Dynamics.')

        runIDTool(pathGenericSetupFileID, ELFile, GRFFile,              
                pathScaledModel, pathIKFile, timeRange, pathOutputFolder,
                filteringFrequency=6)
    
