# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 14:32:04 2023

Downloads data, reprocesses scaling/IK with new model, runs ID.
For NMD data collection

@author: Scott Uhlrich

Edited by Parker
"""

import sys
import os
import glob
repoDir = os.path.abspath('./..')
sys.path.append(repoDir)

import utils
from main import main
from utilsOpenSim import runIDTool, runScaleTool, getScaleTimeRange, runIKTool


# # sessions to process
# sessions = ['057d10da-34c7-4fb7-a127-6040010dde06']
# trial_names = ['brooke']

# # data directory
# # dataDir = 'C:/Users/suhlr/Documents/MyRepositories/sandbox/' # A 'Data' folder will be appended
# dataDir = '..' # A 'Data' folder will be appended
# dataDir = '/Users/psr/Documents/Stanford/Research/Delp/OpenCap/opencap-core'

def fancy_shoulder(session, trial, dataDir, tempDataDir):


    # new model name. If using old model, set to None
    newModelName = 'LaiUhlrich2022_shoulder'
    # newModelName = 'LaiUhlrich2022_shoulder_generic'

    # Lowpass filter frequency for IK results before ID
    ikFiltFreq = 4

    # parts of code to run
    runScaling = True
    runIK = True
    runID = True

    # which IK files to use
    if 'shoulder' in newModelName:
        suffix_model = '_shoulder'
    else:
        suffix_model = ''

    print('Processing session ' + session + '.')
    sessionDir = os.path.join(dataDir,'opencap_data',session)
    opensimDir = os.path.join(sessionDir, 'OpenSimData')
    
    # Download data if not already there
    if not os.path.exists(sessionDir):
        print('Downloading...')
        print(sessionDir)
        return
        utils.downloadAndZipSession(session,justDownload=True,
                                    data_dir=os.path.join(dataDir,'Data'))

    # load metadata
    sessionMetadata = utils.importMetadata(os.path.join(sessionDir,
                                                'sessionMetadata.yaml'))

    if newModelName != None:
        if runScaling:

            # change metadata
            oldModel = sessionMetadata['openSimModel']
            sessionMetadata['openSimModel']= newModelName
            # save metadata
            utils.saveMetadata(sessionMetadata, os.path.join(sessionDir, 
                                                    'sessionMetadata.yaml'))
            
            # re-run scaling
            outputScaledModelDir = os.path.join(opensimDir, 'Model')
            neutralTrialID = utils.getNeutralTrialID(session)
            
            pathTRCFile4Scaling = os.path.join(sessionDir,'MarkerData','PostAugmentation',
                                               'neutral','neutral.trc')
    
            print('Re-scaling model.')
            timeRange4Scaling = getScaleTimeRange(pathTRCFile4Scaling,
                                                  thresholdPosition=0.007,
                                                  thresholdTime=0.1,
                                                  removeRoot=True)     
            
            # Paths to generic files
            genericSetupFile4ScalingName = (
                'Setup_scaling_RajagopalModified2016_withArms_KA.xml')
            pathGenericSetupFile4Scaling = os.path.join(
                repoDir,'opensimPipeline', 'Scaling', genericSetupFile4ScalingName)
            pathGenericModel4Scaling = os.path.join(
                repoDir,'opensimPipeline', 'Models', 
                sessionMetadata['openSimModel'] + '.osim')
            
            # Run scale tool.
            print('Running Scaling')
            pathScaledModel = runScaleTool(
                pathGenericSetupFile4Scaling, pathGenericModel4Scaling,
                sessionMetadata['mass_kg'], pathTRCFile4Scaling, 
                timeRange4Scaling, outputScaledModelDir,
                subjectHeight=sessionMetadata['height_m'])
        else:
            pathScaledModel = os.path.join(opensimDir,'Model',
                                           sessionMetadata['openSimModel'] + '_scaled.osim')
        
        # IK
        for trial_name in trial_names:
            
            outputIKDir = os.path.join(opensimDir, 'Kinematics')
            os.makedirs(outputIKDir, exist_ok=True)
            # Path setup file.
            genericSetupFile4IKName = 'Setup_IK{}.xml'.format(suffix_model)
            pathGenericSetupFile4IK = os.path.join(
                repoDir,'opensimPipeline', 'IK', genericSetupFile4IKName)
            # Path TRC file.
            pathTRCFile4IK = os.path.join(sessionDir,'MarkerData',
                                          'PostAugmentation',trial_name,
                                          trial_name + '.trc')
            # name of output file
            ikFileName = trial_name + suffix_model
            
            # Run IK tool. 
            if runIK:
                print('Re-running IK for ' + trial_name)
                pathOutputIK = runIKTool(
                    pathGenericSetupFile4IK, pathScaledModel, 
                    pathTRCFile4IK, outputIKDir,
                    IKFileName=ikFileName)

        # change metadata back
        if runScaling:
            sessionMetadata ['opensimModel']= oldModel
            utils.saveMetadata(sessionMetadata, os.path.join(sessionDir, 
                                                    'sessionMetadata.yaml'))
        
    else:
        # find oldest model in file
        files = glob.glob(os.path.join(opensimDir, 'Model','*.osim'))
        pathScaledModel = sorted( files, key = lambda file: os.path.getctime(file))[0]
        
    # ID
    if runID:
        for trial in trial_names:
            print('Running ID for ' + trial + '.')
            # run ID
            pathGenericSetupFileID = os.path.join(repoDir,'opensimPipeline','ID', 'Setup_ID.xml')
            pathIKFile = os.path.join(opensimDir, 'Kinematics', trial + suffix_model + '.mot')
            pathOutputFolder = os.path.join(opensimDir, 'Dynamics')
            os.makedirs(pathOutputFolder, exist_ok=True)
            idTrialName = trial + suffix_model
    
            # we don't have a GRF, EL, or time range, so we set it to None
            ELFile = None
            GRFFile = None
            timeRange = None
            
            runIDTool(pathGenericSetupFileID, ELFile, GRFFile,              
                    pathScaledModel, pathIKFile, timeRange, pathOutputFolder,
                    filteringFrequency=ikFiltFreq,IKFileName=idTrialName)
        
