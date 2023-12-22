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
from pathlib import Path
import glob
repoDir = os.path.abspath('./..')
sys.path.append(repoDir)

import utils
from main import main
from utilsOpenSim import runIDTool, runScaleTool, getScaleTimeRange, runIKTool

def fancy_shoulder(session, trial, dataDir):
    newModelName = 'LaiUhlrich2022_shoulder'
    suffix_model = '_shoulder'

    # Lowpass filter frequency for IK results before ID
    ikFiltFreq = 4

    # parts of code to run
    runScaling = True
    runIK = True
    runID = True

    # print('Processing session ' + session + '.')
    sessionDir = Path(dataDir).absolute() / 'opencap_data' / session
    opensimDir = sessionDir / 'OpenSimData'

    # load metadata
    metadataPath = sessionDir / 'sessionMetadata.yaml'
    sessionMetadata = utils.importMetadata(metadataPath)


    ### RUN SCALING ###

    # change metadata
    oldModel = sessionMetadata['openSimModel']
    sessionMetadata['openSimModel'] = newModelName
    utils.saveMetadata(sessionMetadata, metadataPath)
    
    # re-run scaling
    outputScaledModelDir = opensimDir / 'Model'
    # neutralTrialID = utils.getNeutralTrialID(session)
    
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
    

    ### RUN INVERSE KINEMATICS ###

    outputIKDir = os.path.join(opensimDir, 'Kinematics')
    os.makedirs(outputIKDir, exist_ok=True)
    # Path setup file.
    genericSetupFile4IKName = 'Setup_IK{}.xml'.format(suffix_model)
    pathGenericSetupFile4IK = os.path.join(
        repoDir,'opensimPipeline', 'IK', genericSetupFile4IKName)
    # Path TRC file.
    pathTRCFile4IK = os.path.join(sessionDir,'MarkerData',
                                    'PostAugmentation',trial,
                                    trial + '.trc')
    # name of output file
    ikFileName = trial + suffix_model
    
    # Run IK tool. 
    # print('Re-running IK for ' + trial)
    pathOutputIK = runIKTool(
        pathGenericSetupFile4IK, pathScaledModel, 
        pathTRCFile4IK, outputIKDir,
        IKFileName=ikFileName)

    # change metadata back
    sessionMetadata ['opensimModel'] = oldModel
    utils.saveMetadata(sessionMetadata, os.path.join(sessionDir, 
                                            'sessionMetadata.yaml'))
        

    ### RUN INVERSE DYNAMICS ###

    # print('Running ID for ' + trial + '.')
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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('fancy_shoulder')
    parser.add_argument('-s', '--session')
    parser.add_argument('-t', '--trial')
    parser.add_argument('-d', '--datadir')
    args = parser.parse_args()

    fancy_shoulder(args.session, args.trial, args.datadir)

