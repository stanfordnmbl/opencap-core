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


This script computes and combines scalar and timeseries data across activities
for each subject, in order to reproduce the results  in the "OpenCap: 3D human 
movement dynamics from smartphone videos" paper.

Before running the script, you need to download the data from simtk.org/opencap. 
See the README in this folder for more details.
"""

import os
import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')

repoDir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../'))
dataDir = os.path.join(repoDir, 'Data')

sys.path.append(repoDir) # utilities from base repository directory
sys.path.append(os.path.join(repoDir,'DataProcessing')) # utilities in child directory

from sklearn.metrics import mean_squared_error, mean_absolute_error
from itertools import combinations
from utilsDataPostprocessing import (segmentSquats, segmentWalkStance, segmentDJ,
                                     getIndsFromTimes, calc_LSI, interpolateNumpyArray)


# %% Settings
fieldStudy = False # True to process field study results, false to process LabValidation results
saveResults = True 


if fieldStudy:
    subjects = ['subject' + str(sub) for sub in range(100)]
    motion_types = motion_types = ['squats','squatsAsym']
    
    poseDetector = 'OpenPose'
    cameraSetup = 'all-cameras'
    data_type = 'Video' 
    modalityFolderName = '' # no Video folder name
    invert_left_right = False
    
else:
    subjects = ['subject' + str(sub) for sub in range(2,12)]
    motion_types = motion_types = ['DJ', 'DJAsym', 'walking', 'walkingTS', 
                                   'squats','squatsAsym','STS','STSweakLegs']

    # Likely fixed settings
    data_type = 'Video' # only set up for video now
    modalityFolderName = data_type
    poseDetector = 'HRNet'
    cameraSetup = '2-cameras'
    invert_left_right = False
    
N = 101 # for interpolation


#%% End user inputs

for iSub,subject in enumerate(subjects):

    results_con = {}
    print('Processing subject {} of {}: '.format(iSub+1,len(subjects)) + subject)
    for iMotion, motion_type in enumerate(motion_types):
       
        if 'DJ' in motion_type:
            align_signals = True
        else:
            align_signals = False
        
           
        if fieldStudy:
            osDir = os.path.join(dataDir,'FieldStudy', subject, 'OpenSimData')
            pathOSData = os.path.join(osDir,'Dynamics')
            
            # Check for data folder
            if not os.path.isdir(pathOSData):
                raise Exception('The data is not found in ' + dataDir + '. Download it from https://simtk.org/projects/opencap, and save to the the repository directory. E.g., Data/FieldStudy')
            
            results = np.load(os.path.join(pathOSData,
                                       '{}_results.npy'.format(motion_type)),
                          allow_pickle=True).item()
        else:
            osDir = os.path.join(dataDir,'LabValidation', subject, 'OpenSimData')
            pathOSData = os.path.join(osDir, modalityFolderName, poseDetector, 
                                      cameraSetup,'Dynamics')
            
            # Check for data folder
            if not os.path.isdir(pathOSData):
                raise Exception('The data is not found in ' + dataDir + '. Download it from https://simtk.org/projects/opencap, and save to the the repository directory. E.g., Data/LabValidation')
            
            results = np.load(os.path.join(pathOSData, 
                                       '{}_results.npy'.format(motion_type)),
                          allow_pickle=True).item()
            
        tempKeys = list(results.keys())
        cases = list(results[tempKeys[0]].keys())


        # %% Select cases and distinguish between mocap and video
        results_sel = {}
        results_sel['mocap'] = {}
        results_sel['video'] = {}
        for variable_type in results:
            results_sel['mocap'][variable_type] = {}
            results_sel['video'][variable_type] = {}
            nMocapCases = 0
            nMobilecapCases = 0
            for case in cases:        
                # Mocap cases        
                if data_type == 'mocap':
                # if cases_toPlot[case]['data_type'] == 'mocap':             
                    results_sel['mocap'][variable_type][case] = (
                        results[variable_type][case])        
                    nMocapCases += 1
                # video cases
                elif data_type == 'Video':
                # elif cases_toPlot[case]['data_type'] == 'Video':       
                    results_sel['video'][variable_type][case] = (
                        results[variable_type][case])    
                    nMobilecapCases += 1
                else:
                    raise ValueError("Unknown type")
        # %% TODO this should be a host of functions that segment all of the data by data type
        selInds = {}
        segSource = 'ref'
        if fieldStudy:
            segSource = 'sim'
        for case in cases:
            # Walking - identify first L toe-off, first L HS, second L toe-off. add swing to end.
            if 'walking' in motion_type:
                forcesFilePath = osDir = os.path.join(dataDir, 'LabValidation',subject, 'ForceData',case.replace('_videoAndMocap','') + '_forces.mot')
                _, sfTimes = segmentWalkStance(forcesFilePath)
                resTimeIdx = results_sel['video']['positions'][case]['headers'].index('time')
                timeVec = results_sel['video']['positions'][case][segSource][resTimeIdx,:]
                sfInds = getIndsFromTimes(sfTimes,timeVec)
                selInds[case] = np.arange(sfInds[0],sfInds[1]+1)
                
              
            elif 'DJ' in motion_type:
                forcesFilePath = osDir = os.path.join(dataDir, 'LabValidation',subject, 'ForceData',case.replace('_videoAndMocap','') + '_forces.mot')
                _, sfTimes = segmentDJ(forcesFilePath)
                resTimeIdx = results_sel['video']['positions'][case]['headers'].index('time')
                timeVec = results_sel['video']['positions'][case][segSource][resTimeIdx,:]
                sfInds = getIndsFromTimes(sfTimes,timeVec)
                selInds[case] = np.arange(sfInds[0],sfInds[1]+1)
               
            elif 'squat' in motion_type:
                resTimeIdx = results_sel['video']['positions'][case]['headers'].index('time')
                pelvTyIdx = results_sel['video']['positions'][case]['headers'].index('pelvis_ty')
                timeVec = results_sel['video']['positions'][case][segSource][resTimeIdx,:]
                pelvis_ty = results_sel['video']['positions'][case][segSource][pelvTyIdx,:]
                if fieldStudy:
                    sfInds = []
                    sfInds.append([0, pelvis_ty.shape[0]-1]) 
                else: 
                    sfInds, _ = segmentSquats(None , pelvis_ty=pelvis_ty , timeVec=timeVec)
                selInds[case] = np.arange(sfInds[0][0],sfInds[0][1]+1)
            elif 'STS' in motion_type:
                resTimeIdx = results_sel['video']['positions'][case]['headers'].index('time')
                pelvTyIdx = results_sel['video']['positions'][case]['headers'].index('pelvis_ty')
                timeVec = results_sel['video']['positions'][case][segSource][resTimeIdx,:]
                pelvis_ty = results_sel['video']['positions'][case][segSource][pelvTyIdx,:]
                sfInds = []
                sfInds.append([0, pelvis_ty.shape[0]-1])                
                selInds[case] = np.arange(sfInds[0][0],sfInds[0][1]+1)                
            else:
                raise Exception('Motion type:' + motion_type + ' not supported')
                    
        #%% Trim to only the segmented portions results_sel
        c_type = 'video'
        for case in cases:
            for variable_type in results:
                if variable_type == 'GRFs_BW_peaks' or variable_type == 'GRFs_peaks':
                    continue
                if 'toTrack' in results_sel[c_type][variable_type][case]:
                    results_sel[c_type][variable_type][case]['toTrack'] = (results_sel[c_type][variable_type][case]['toTrack'][:,selInds[case]])              
                results_sel[c_type][variable_type][case]['ref'] = (results_sel[c_type][variable_type][case]['ref'][:,selInds[case]])
                results_sel[c_type][variable_type][case]['sim'] = (results_sel[c_type][variable_type][case]['sim'][:,selInds[case]])
        
        # %% Interpolate
        results_int = {}
        results_int['mocap'] = {}
        results_int['video'] = {}
        for variable_type in results:
            if variable_type == 'GRFs_BW_peaks' or variable_type == 'GRFs_peaks':
                continue
            results_int['mocap'][variable_type] = {}
            results_int['video'][variable_type] = {}
            for case in cases:        
                # Mocap cases        
                if data_type == 'mocap':
                # if cases_toPlot[case]['data_type'] == 'mocap':  
                    c_type = 'mocap'
                elif data_type == 'Video':
                # elif cases_toPlot[case]['data_type'] == 'Video':
                    c_type = 'video'
                else:
                    raise ValueError("Unknown type")
                results_int[c_type][variable_type][case] = {}            
                results_int[c_type][variable_type][case]['headers'] = (
                    results_sel[c_type][variable_type][case]['headers'])
                results_int[c_type][variable_type][case]['sim'] = (
                    interpolateNumpyArray(
                        results_sel[c_type][variable_type][case]['sim'], N))
                results_int[c_type][variable_type][case]['ref'] = (
                    interpolateNumpyArray(
                        results_sel[c_type][variable_type][case]['ref'], N))
                if 'toTrack' in results_sel[c_type][variable_type][case]:
                    results_int[c_type][variable_type][case]['toTrack'] = (
                        interpolateNumpyArray(
                            results_sel[c_type][variable_type][case]['toTrack'], N))     
                if 'so' in results_sel[c_type][variable_type][case]:
                    results_int[c_type][variable_type][case]['so'] = (
                        interpolateNumpyArray(
                            results_sel[c_type][variable_type][case]['so'], N)) 
                
                    
        # %% Concatenate
        results_con[motion_type] = {}
        results_con[motion_type]['mocap'] = {}
        results_con[motion_type]['video'] = {}
        
        for variable_type in results:
            if variable_type == 'GRFs_BW_peaks' or variable_type == 'GRFs_peaks':
                continue
            results_con[motion_type]['mocap'][variable_type] = {}
            results_con[motion_type]['video'][variable_type] = {}
            
            nRows = results_int['video'][variable_type][cases[0]]['ref'].shape[0]
            results_con[motion_type]['mocap'][variable_type]['ref'] = np.zeros((nRows, N, nMocapCases))
            results_con[motion_type]['mocap'][variable_type]['sim'] = np.zeros((nRows, N, nMocapCases))    
            results_con[motion_type]['video'][variable_type]['ref'] = np.zeros((nRows, N, nMobilecapCases))
            results_con[motion_type]['video'][variable_type]['sim'] = np.zeros((nRows, N, nMobilecapCases))
            
            if 'toTrack' in results_int['video'][variable_type][cases[0]]:
                results_con[motion_type]['video'][variable_type]['toTrack'] = np.zeros((nRows, N, nMobilecapCases))
            if cases[0] in results_int['mocap'][variable_type]:
                if 'toTrack' in results_int['mocap'][variable_type][cases[0]]:
                    results_con[motion_type]['mocap'][variable_type]['toTrack'] = np.zeros((nRows, N, nMobilecapCases))
            if 'so' in results_int['video'][variable_type][cases[0]]:
                results_con[motion_type]['video'][variable_type]['so'] = np.zeros((nRows, N, nMobilecapCases))
            if cases[0] in results_int['mocap'][variable_type]:
                if 'so' in results_int['mocap'][variable_type][cases[0]]:
                    results_con[motion_type]['mocap'][variable_type]['so'] = np.zeros((nRows, N, nMobilecapCases))
                
            
            c_mocap = 0
            c_mobilecap = 0
            for case in cases: 
                if data_type == 'mocap':
                # if cases_toPlot[case]['data_type'] == 'mocap':  
                    c_type = 'mocap'
                    if c_mocap == 0:
                        results_con[motion_type][c_type][variable_type]['headers'] = results_int[c_type][variable_type][case]['headers']  
                    results_con[motion_type][c_type][variable_type]['ref'][:,:,c_mocap] = results_int[c_type][variable_type][case]['ref']
                    results_con[motion_type][c_type][variable_type]['sim'][:,:,c_mocap] = results_int[c_type][variable_type][case]['sim']
                    if 'toTrack' in results_int[c_type][variable_type][case]:
                        results_con[motion_type][c_type][variable_type]['toTrack'][:,:,c_mocap] = results_int[c_type][variable_type][case]['toTrack']
                    if 'so' in results_int[c_type][variable_type][case]:
                        results_con[motion_type][c_type][variable_type]['so'][:,:,c_mocap] = results_int[c_type][variable_type][case]['so']
                    c_mocap += 1
                elif data_type == 'Video':
                # elif cases_toPlot[case]['data_type'] == 'Video':
                    c_type = 'video'
                    if c_mobilecap == 0:
                        results_con[motion_type][c_type][variable_type]['headers'] = results_int[c_type][variable_type][case]['headers']            
                    results_con[motion_type][c_type][variable_type]['ref'][:,:,c_mobilecap] = results_int[c_type][variable_type][case]['ref']
                    results_con[motion_type][c_type][variable_type]['sim'][:,:,c_mobilecap] = results_int[c_type][variable_type][case]['sim']
                    if 'toTrack' in results_int[c_type][variable_type][case]:
                        results_con[motion_type][c_type][variable_type]['toTrack'][:,:,c_mobilecap] = results_int[c_type][variable_type][case]['toTrack']
                    if 'so' in results_int[c_type][variable_type][case]:
                        results_con[motion_type][c_type][variable_type]['so'][:,:,c_mobilecap] = results_int[c_type][variable_type][case]['so']
                    c_mobilecap += 1
                else:
                    raise ValueError("Unknown type")
                    
        
        # %% Mean and std
        for variable_type in results:
            if variable_type == 'GRFs_BW_peaks' or variable_type == 'GRFs_peaks':
                continue
            if results_con[motion_type]['mocap'][variable_type]['ref'].size != 0:
                results_con[motion_type]['mocap'][variable_type]['ref_mean'] = np.mean(results_con[motion_type]['mocap'][variable_type]['ref'], axis=2)
                results_con[motion_type]['mocap'][variable_type]['sim_mean'] = np.mean(results_con[motion_type]['mocap'][variable_type]['sim'], axis=2)
                results_con[motion_type]['mocap'][variable_type]['ref_std'] = np.std(results_con[motion_type]['mocap'][variable_type]['ref'], axis=2)
                results_con[motion_type]['mocap'][variable_type]['sim_std'] = np.std(results_con[motion_type]['mocap'][variable_type]['sim'], axis=2)
            
            if results_con[motion_type]['video'][variable_type]['ref'].size != 0:
                results_con[motion_type]['video'][variable_type]['ref_mean'] = np.mean(results_con[motion_type]['video'][variable_type]['ref'], axis=2)
                results_con[motion_type]['video'][variable_type]['sim_mean'] = np.mean(results_con[motion_type]['video'][variable_type]['sim'], axis=2)
                results_con[motion_type]['video'][variable_type]['ref_std'] = np.std(results_con[motion_type]['video'][variable_type]['ref'], axis=2)
                results_con[motion_type]['video'][variable_type]['sim_std'] = np.std(results_con[motion_type]['video'][variable_type]['sim'], axis=2)

            if 'so' in results_con[motion_type]['video'][variable_type]:
                results_con[motion_type]['video'][variable_type]['so_mean'] = np.mean(results_con[motion_type]['video'][variable_type]['so'], axis=2)
                results_con[motion_type]['video'][variable_type]['so_std'] = np.std(results_con[motion_type]['video'][variable_type]['so'], axis=2)
                
        # %% TODO compute motionType-dependent scalars
        results_con[motion_type]['video']['scalars'] = {}
        
        if 'walking' in motion_type:
            # peak KAM
            idx_KAM_l = [results_con[motion_type]['video']['KAMs_BWht']['headers'].index('KAM_l')]
            results_con[motion_type]['video']['scalars']['ref_peakKAM_l'] = np.max(
                results_con[motion_type]['video']['KAMs_BWht']['ref'][idx_KAM_l,:50,:],axis=1)
            results_con[motion_type]['video']['scalars']['sim_peakKAM_l'] = np.max(
                results_con[motion_type]['video']['KAMs_BWht']['sim'][idx_KAM_l,:50,:],axis=1)
            
            idx_KAM_l = [results_con[motion_type]['video']['KAMs_BWht']['headers'].index('KAM_l')]
            results_con[motion_type]['video']['scalars']['ref_meanKAM_l'] = np.mean(
                results_con[motion_type]['video']['KAMs_BWht']['ref'][idx_KAM_l,:,:],axis=1)
            results_con[motion_type]['video']['scalars']['sim_meanKAM_l'] = np.mean(
                results_con[motion_type]['video']['KAMs_BWht']['sim'][idx_KAM_l,:,:],axis=1)
            
            idx = [results_con[motion_type]['video']['KAMs_BWht']['headers'].index('KAM_r')]
            results_con[motion_type]['video']['scalars']['ref_peakKAM_r'] = np.max(
                results_con[motion_type]['video']['KAMs_BWht']['ref'][idx,:50,:],axis=1)
            results_con[motion_type]['video']['scalars']['sim_peakKAM_r'] = np.max(
                results_con[motion_type]['video']['KAMs_BWht']['sim'][idx,:50,:],axis=1)
            
            idx = [results_con[motion_type]['video']['KAMs_BWht']['headers'].index('KAM_r')]
            results_con[motion_type]['video']['scalars']['ref_meanKAM_r'] = np.mean(
                results_con[motion_type]['video']['KAMs_BWht']['ref'][idx_KAM_l,:,:],axis=1)
            results_con[motion_type]['video']['scalars']['sim_meanKAM_r'] = np.mean(
                results_con[motion_type]['video']['KAMs_BWht']['sim'][idx_KAM_l,:,:],axis=1)
            
            idx_KAM_l = [results_con[motion_type]['video']['KAMs_BWht']['headers'].index('KAM_l')]
            results_con[motion_type]['video']['scalars']['ref_peak2KAM_l'] = np.max(
                results_con[motion_type]['video']['KAMs_BWht']['ref'][idx_KAM_l,50:,:],axis=1)
            results_con[motion_type]['video']['scalars']['sim_peak2KAM_l'] = np.max(
                results_con[motion_type]['video']['KAMs_BWht']['sim'][idx_KAM_l,50:,:],axis=1)
            
            idx = [results_con[motion_type]['video']['KAMs_BWht']['headers'].index('KAM_r')]
            results_con[motion_type]['video']['scalars']['ref_peakKAM_r'] = np.max(
                results_con[motion_type]['video']['KAMs_BWht']['ref'][idx,50:,:],axis=1)
            results_con[motion_type]['video']['scalars']['sim_peakKAM_r'] = np.max(
                results_con[motion_type]['video']['KAMs_BWht']['sim'][idx,50:,:],axis=1)
            
            # MCF
            idx = [results_con[motion_type]['video']['MCFs_BW']['headers'].index('MCF_l')]
            results_con[motion_type]['video']['scalars']['ref_peakMCF_l'] = np.max(
                results_con[motion_type]['video']['MCFs_BW']['ref'][idx,:50,:],axis=1)
            results_con[motion_type]['video']['scalars']['sim_peakMCF_l'] = np.max(
                results_con[motion_type]['video']['MCFs_BW']['sim'][idx,:50,:],axis=1)
            
            idx = [results_con[motion_type]['video']['MCFs_BW']['headers'].index('MCF_r')]
            results_con[motion_type]['video']['scalars']['ref_peakMCF_r'] = np.max(
                results_con[motion_type]['video']['MCFs_BW']['ref'][idx,:50,:],axis=1)
            results_con[motion_type]['video']['scalars']['sim_peakMCF_r'] = np.max(
                results_con[motion_type]['video']['MCFs_BW']['sim'][idx,:50,:],axis=1)
            
            idx = [results_con[motion_type]['video']['MCFs_BW']['headers'].index('MCF_l')]
            results_con[motion_type]['video']['scalars']['ref_peak2MCF_l'] = np.max(
                results_con[motion_type]['video']['MCFs_BW']['ref'][idx,50:,:],axis=1)
            results_con[motion_type]['video']['scalars']['sim_peak2MCF_l'] = np.max(
                results_con[motion_type]['video']['MCFs_BW']['sim'][idx,50:,:],axis=1)
            
            idx = [results_con[motion_type]['video']['MCFs_BW']['headers'].index('MCF_r')]
            results_con[motion_type]['video']['scalars']['ref_peak2MCF_r'] = np.max(
                results_con[motion_type]['video']['MCFs_BW']['ref'][idx,50:,:],axis=1)
            results_con[motion_type]['video']['scalars']['sim_peak2MCF_r'] = np.max(
                results_con[motion_type]['video']['MCFs_BW']['sim'][idx,50:,:],axis=1)
                            
            idx = [results_con[motion_type]['video']['MCFs_BW']['headers'].index('MCF_l')]
            results_con[motion_type]['video']['scalars']['ref_meanMCF_l'] = np.mean(
                results_con[motion_type]['video']['MCFs_BW']['ref'][idx,:,:],axis=1)
            results_con[motion_type]['video']['scalars']['sim_meanMCF_l'] = np.mean(
                results_con[motion_type]['video']['MCFs_BW']['sim'][idx,:,:],axis=1)
            
            idx = [results_con[motion_type]['video']['MCFs_BW']['headers'].index('MCF_r')]
            results_con[motion_type]['video']['scalars']['ref_meanMCF_r'] = np.mean(
                results_con[motion_type]['video']['MCFs_BW']['ref'][idx,:,:],axis=1)
            results_con[motion_type]['video']['scalars']['sim_meanMCF_r'] = np.mean(
                results_con[motion_type]['video']['MCFs_BW']['sim'][idx,:,:],axis=1)
            
            idx = [results_con[motion_type]['video']['MCFs']['headers'].index('MCF_l')]
            results_con[motion_type]['video']['scalars']['ref_peakMCFnonNorm_l'] = np.max(
                results_con[motion_type]['video']['MCFs']['ref'][idx,:50,:],axis=1)
            results_con[motion_type]['video']['scalars']['sim_peakMCFnonNorm_l'] = np.max(
                results_con[motion_type]['video']['MCFs']['sim'][idx,:50,:],axis=1)
                          
            idx = [results_con[motion_type]['video']['MCFs']['headers'].index('MCF_r')]
            results_con[motion_type]['video']['scalars']['ref_peakMCFnonNorm_r'] = np.max(
                results_con[motion_type]['video']['MCFs']['ref'][idx,:50,:],axis=1)
            results_con[motion_type]['video']['scalars']['sim_peakMCFnonNorm_r'] = np.max(
                results_con[motion_type]['video']['MCFs']['sim'][idx,:50,:],axis=1)
            
            idx = [results_con[motion_type]['video']['MCFs']['headers'].index('MCF_l')]
            results_con[motion_type]['video']['scalars']['ref_peak2MCFnonNorm_l'] = np.max(
                results_con[motion_type]['video']['MCFs']['ref'][idx,50:,:],axis=1)
            results_con[motion_type]['video']['scalars']['sim_peak2MCFnonNorm_l'] = np.max(
                results_con[motion_type]['video']['MCFs']['sim'][idx,50:,:],axis=1)
                          
            idx = [results_con[motion_type]['video']['MCFs']['headers'].index('MCF_r')]
            results_con[motion_type]['video']['scalars']['ref_peak2MCFnonNorm_r'] = np.max(
                results_con[motion_type]['video']['MCFs']['ref'][idx,50:,:],axis=1)
            results_con[motion_type]['video']['scalars']['sim_peak2MCFnonNorm_r'] = np.max(
                results_con[motion_type]['video']['MCFs']['sim'][idx,50:,:],axis=1)
            
            # Glute Med muscle activation
            idx_glmed_l = [results_con[motion_type]['video']['activations']['headers'].index('glmed1_l')]
            results_con[motion_type]['video']['scalars']['ref_avgGmedAct'] = np.mean(
                results_con[motion_type]['video']['activations']['ref'][idx_glmed_l,0:50,:],axis=1)
            results_con[motion_type]['video']['scalars']['sim_avgGmedAct'] = np.mean(
                results_con[motion_type]['video']['activations']['sim'][idx_glmed_l,0:50,:],axis=1)
            
        elif 'DJ' in motion_type:
            # peak KAM_l maximum of the negative external adduction moment during first 50%
            idx_KAM_l = [results_con[motion_type]['video']['KAMs']['headers'].index('KAM_l')]
            results_con[motion_type]['video']['scalars']['ref_peakKAM_l'] = np.max(
                -results_con[motion_type]['video']['KAMs']['ref'][idx_KAM_l,0:50,:],axis=1)
            results_con[motion_type]['video']['scalars']['sim_peakKAM_l'] = np.max(
                -results_con[motion_type]['video']['KAMs']['sim'][idx_KAM_l,0:50,:],axis=1)
            
            # peak KAM_r
            idx_KAM_r = [results_con[motion_type]['video']['KAMs']['headers'].index('KAM_r')]
            results_con[motion_type]['video']['scalars']['ref_peakKAM_r'] = np.max(
                -results_con[motion_type]['video']['KAMs']['ref'][idx_KAM_r,0:50,:],axis=1)
            results_con[motion_type]['video']['scalars']['sim_peakKAM_r'] = np.max(
                -results_con[motion_type]['video']['KAMs']['sim'][idx_KAM_r,0:50,:],axis=1)      
            
            # KAM Asymmetry
            results_con[motion_type]['video']['scalars']['ref_peakKAM_Asym'] = calc_LSI(results_con[
                motion_type]['video']['scalars']['ref_peakKAM_l'] , results_con[
                motion_type]['video']['scalars']['ref_peakKAM_r'])
            results_con[motion_type]['video']['scalars']['sim_peakKAM_Asym'] = calc_LSI(
                results_con[motion_type]['video']['scalars']['sim_peakKAM_l'],
                results_con[motion_type]['video']['scalars']['sim_peakKAM_r'])
                    
            # peak KEM_l
            idx_KEM_l = [results_con[motion_type]['video']['torques_BWht']['headers'].index('knee_angle_l')]
            results_con[motion_type]['video']['scalars']['ref_peakKEM_l'] = np.max(-
                results_con[motion_type]['video']['torques_BWht']['ref'][idx_KEM_l,:,:],axis=1)
            results_con[motion_type]['video']['scalars']['sim_peakKEM_l'] = np.max(-
                results_con[motion_type]['video']['torques_BWht']['sim'][idx_KEM_l,:,:],axis=1)
            
            # peak KEM_r
            idx_KEM_r = [results_con[motion_type]['video']['torques_BWht']['headers'].index('knee_angle_r')]
            results_con[motion_type]['video']['scalars']['ref_peakKEM_r'] = np.max(-
                results_con[motion_type]['video']['torques_BWht']['ref'][idx_KEM_r,:,:],axis=1)
            results_con[motion_type]['video']['scalars']['sim_peakKEM_r'] = np.max(-
                results_con[motion_type]['video']['torques_BWht']['sim'][idx_KEM_r,:,:],axis=1)
            
            # peak KEM Asymmetry
            results_con[motion_type]['video']['scalars']['ref_peakKEM_Asym'] = calc_LSI(results_con[
                motion_type]['video']['scalars']['ref_peakKEM_l'], results_con[
                motion_type]['video']['scalars']['ref_peakKEM_r'])
            results_con[motion_type]['video']['scalars']['sim_peakKEM_Asym'] = calc_LSI(results_con[
                motion_type]['video']['scalars']['sim_peakKEM_l'] , results_con[
                motion_type]['video']['scalars']['sim_peakKEM_r'])
                    
            # mean KEM_l
            idx_KEM_l = [results_con[motion_type]['video']['torques_BWht']['headers'].index('knee_angle_l')]
            results_con[motion_type]['video']['scalars']['ref_meanKEM_l'] = np.mean(-
                results_con[motion_type]['video']['torques_BWht']['ref'][idx_KEM_l,:,:],axis=1)
            results_con[motion_type]['video']['scalars']['sim_meanKEM_l'] = np.mean(-
                results_con[motion_type]['video']['torques_BWht']['sim'][idx_KEM_l,:,:],axis=1)
            
            # mean KEM_r
            idx_KEM_r = [results_con[motion_type]['video']['torques_BWht']['headers'].index('knee_angle_r')]
            results_con[motion_type]['video']['scalars']['ref_meanKEM_r'] = np.mean(-
                results_con[motion_type]['video']['torques_BWht']['ref'][idx_KEM_r,:,:],axis=1)
            results_con[motion_type]['video']['scalars']['sim_meanKEM_r'] = np.mean(-
                results_con[motion_type]['video']['torques_BWht']['sim'][idx_KEM_r,:,:],axis=1)
            
            # mean KEM Asymmetry
            results_con[motion_type]['video']['scalars']['ref_meanKEM_Asym'] = calc_LSI(results_con[
                motion_type]['video']['scalars']['ref_meanKEM_l'] , results_con[
                motion_type]['video']['scalars']['ref_meanKEM_r'])
            results_con[motion_type]['video']['scalars']['sim_meanKEM_Asym'] = calc_LSI(results_con[
                motion_type]['video']['scalars']['sim_meanKEM_l'] , results_con[
                motion_type]['video']['scalars']['sim_meanKEM_r'])
                    
            # peak vGRF_l
            idx = [results_con[motion_type]['video']['GRFs_BW']['headers'].index('ground_force_l_vy')]
            results_con[motion_type]['video']['scalars']['ref_peakvGRF_l'] = np.max(
                results_con[motion_type]['video']['GRFs_BW']['ref'][idx,:,:],axis=1)
            results_con[motion_type]['video']['scalars']['sim_peakvGRF_l'] = np.max(
                results_con[motion_type]['video']['GRFs_BW']['sim'][idx,:,:],axis=1)
            
            # peak vGRF_r
            idx = [results_con[motion_type]['video']['GRFs_BW']['headers'].index('ground_force_r_vy')]
            results_con[motion_type]['video']['scalars']['ref_peakvGRF_r'] = np.max(
                results_con[motion_type]['video']['GRFs_BW']['ref'][idx,:,:],axis=1)
            results_con[motion_type]['video']['scalars']['sim_peakvGRF_r'] = np.max(
                results_con[motion_type]['video']['GRFs_BW']['sim'][idx,:,:],axis=1)
            
            # peak vGRF Asymmetry
            results_con[motion_type]['video']['scalars']['ref_peakvGRF_Asym'] = calc_LSI(results_con[
                motion_type]['video']['scalars']['ref_peakvGRF_l'] , results_con[
                motion_type]['video']['scalars']['ref_peakvGRF_r'])
            results_con[motion_type]['video']['scalars']['sim_peakvGRF_Asym'] = calc_LSI(results_con[
                motion_type]['video']['scalars']['sim_peakvGRF_l'] , results_con[
                motion_type]['video']['scalars']['sim_peakvGRF_r'])
                    
            # mean vGRF_l
            idx = [results_con[motion_type]['video']['GRFs_BW']['headers'].index('ground_force_l_vy')]
            results_con[motion_type]['video']['scalars']['ref_meanvGRF_l'] = np.mean(
                results_con[motion_type]['video']['GRFs_BW']['ref'][idx,:,:],axis=1)
            results_con[motion_type]['video']['scalars']['sim_meanvGRF_l'] = np.mean(
                results_con[motion_type]['video']['GRFs_BW']['sim'][idx,:,:],axis=1)
            
            # mean vGRF_r
            idx = [results_con[motion_type]['video']['GRFs_BW']['headers'].index('ground_force_r_vy')]
            results_con[motion_type]['video']['scalars']['ref_meanvGRF_r'] = np.mean(
                results_con[motion_type]['video']['GRFs_BW']['ref'][idx,:,:],axis=1)
            results_con[motion_type]['video']['scalars']['sim_meanvGRF_r'] = np.mean(
                results_con[motion_type]['video']['GRFs_BW']['sim'][idx,:,:],axis=1)
            
            # mean vGRF Asymmetry
            results_con[motion_type]['video']['scalars']['ref_meanvGRF_Asym'] = calc_LSI(results_con[
                motion_type]['video']['scalars']['ref_meanvGRF_l'] , results_con[
                motion_type]['video']['scalars']['ref_meanvGRF_r'])
            results_con[motion_type]['video']['scalars']['sim_meanvGRF_Asym'] = calc_LSI(results_con[
                motion_type]['video']['scalars']['sim_meanvGRF_l'] , results_con[
                motion_type]['video']['scalars']['sim_meanvGRF_r'])
                    
            # KFA_ROM
            idx = [results_con[motion_type]['video']['positions']['headers'].index('knee_angle_l')]
            results_con[motion_type]['video']['scalars']['ref_KFA_ROM_l'] = np.ptp(
                results_con[motion_type]['video']['positions']['ref'][idx,:,:],axis=1)
            results_con[motion_type]['video']['scalars']['sim_KFA_ROM_l'] = np.ptp(
                results_con[motion_type]['video']['positions']['sim'][idx,:,:],axis=1)
            
            # KFA_ROM
            idx = [results_con[motion_type]['video']['positions']['headers'].index('knee_angle_r')]
            results_con[motion_type]['video']['scalars']['ref_KFA_ROM_r'] = np.ptp(
                results_con[motion_type]['video']['positions']['ref'][idx,:,:],axis=1)
            results_con[motion_type]['video']['scalars']['sim_KFA_ROM_r'] = np.ptp(
                results_con[motion_type]['video']['positions']['sim'][idx,:,:],axis=1)
            
            # peak KFA
            idx = [results_con[motion_type]['video']['positions']['headers'].index('knee_angle_l')]
            results_con[motion_type]['video']['scalars']['ref_peakKFA_l'] = np.max(
                results_con[motion_type]['video']['positions']['ref'][idx,:,:],axis=1)
            results_con[motion_type]['video']['scalars']['sim_peakKFA_l'] = np.max(
                results_con[motion_type]['video']['positions']['sim'][idx,:,:],axis=1)
            
            # peak KFA
            idx = [results_con[motion_type]['video']['positions']['headers'].index('knee_angle_r')]
            results_con[motion_type]['video']['scalars']['ref_peakKFA_r'] = np.max(
                results_con[motion_type]['video']['positions']['ref'][idx,:,:],axis=1)
            results_con[motion_type]['video']['scalars']['sim_peakKFA_r'] = np.max(
                results_con[motion_type]['video']['positions']['sim'][idx,:,:],axis=1)
            
            # VL_l
            idx = [results_con[motion_type]['video']['activations']['headers'].index('vasmed_l')]
            results_con[motion_type]['video']['scalars']['ref_meanVLAct_l'] = np.mean(
                results_con[motion_type]['video']['activations']['ref'][idx,:,:],axis=1)
            results_con[motion_type]['video']['scalars']['sim_meanVLAct_l'] = np.mean(
                results_con[motion_type]['video']['activations']['sim'][idx,:,:],axis=1)
            
            # VL_r
            idx = [results_con[motion_type]['video']['activations']['headers'].index('vasmed_r')]
            results_con[motion_type]['video']['scalars']['ref_meanVLAct_r'] = np.mean(
                results_con[motion_type]['video']['activations']['ref'][idx,:,:],axis=1)
            results_con[motion_type]['video']['scalars']['sim_meanVLAct_r'] = np.mean(
                results_con[motion_type]['video']['activations']['sim'][idx,:,:],axis=1)
            
            # VL_Asym
            results_con[motion_type]['video']['scalars']['ref_meanVLAct_Asym'] = calc_LSI(results_con[
                motion_type]['video']['scalars']['ref_meanVLAct_l'] , results_con[
                motion_type]['video']['scalars']['ref_meanVLAct_r'])
            results_con[motion_type]['video']['scalars']['sim_meanVLAct_Asym'] = calc_LSI(results_con[
                motion_type]['video']['scalars']['sim_meanVLAct_l'] , results_con[
                motion_type]['video']['scalars']['sim_meanVLAct_r'])
            
        elif 'squat' in motion_type:
            actSuffix = '' 
            # mean(VL,VM)_l
            idx_VL = [results_con[motion_type]['video']['activations' + actSuffix]['headers'].index('vaslat_l')]
            idx_VM = [results_con[motion_type]['video']['activations' + actSuffix]['headers'].index('vasmed_l')]
            results_con[motion_type]['video']['scalars']['ref_meanVLVMAct_l'] = np.mean(np.mean(
                results_con[motion_type]['video']['activations' + actSuffix]['ref'][[idx_VL,idx_VM],:,:],axis=0),axis=1)
            results_con[motion_type]['video']['scalars']['sim_meanVLVMAct_l'] = np.mean(np.mean(
                results_con[motion_type]['video']['activations' + actSuffix]['sim'][[idx_VL,idx_VM],:,:],axis=0),axis=1)
            if not fieldStudy:
                results_con[motion_type]['video']['scalars']['so_meanVLVMAct_l'] = np.mean(np.mean(
                    results_con[motion_type]['video']['activations']['so'][[idx_VL,idx_VM],:,:],axis=0),axis=1)
            
            # mean(VL,VM)_r
            idx_VL = [results_con[motion_type]['video']['activations' + actSuffix]['headers'].index('vaslat_r')]
            idx_VM = [results_con[motion_type]['video']['activations' + actSuffix]['headers'].index('vasmed_r')]
            results_con[motion_type]['video']['scalars']['ref_meanVLVMAct_r'] = np.mean(np.mean(
                results_con[motion_type]['video']['activations' + actSuffix]['ref'][[idx_VL,idx_VM],:,:],axis=0),axis=1)
            results_con[motion_type]['video']['scalars']['sim_meanVLVMAct_r'] = np.mean(np.mean(
                results_con[motion_type]['video']['activations' + actSuffix]['sim'][[idx_VL,idx_VM],:,:],axis=0),axis=1)
            if not fieldStudy:
                results_con[motion_type]['video']['scalars']['so_meanVLVMAct_r'] = np.mean(np.mean(
                    results_con[motion_type]['video']['activations']['so'][[idx_VL,idx_VM],:,:],axis=0),axis=1)
            
            # mean KEM_both
            results_con[motion_type]['video']['scalars']['ref_meanVLVMAct_both'] = np.mean(np.concatenate((
                results_con[motion_type]['video']['scalars']['ref_meanVLVMAct_r'],
                results_con[motion_type]['video']['scalars']['ref_meanVLVMAct_l'])),axis=0,keepdims=True)
            results_con[motion_type]['video']['scalars']['sim_meanVLVMAct_both'] = np.mean(np.concatenate((
                results_con[motion_type]['video']['scalars']['sim_meanVLVMAct_r'],
                results_con[motion_type]['video']['scalars']['sim_meanVLVMAct_l'])),axis=0,keepdims=True)
            
            # mean(VL,VM) Asym
            results_con[motion_type]['video']['scalars']['ref_meanVLVMAct_Asym'] = calc_LSI(results_con[
                motion_type]['video']['scalars']['ref_meanVLVMAct_l'] , results_con[
                motion_type]['video']['scalars']['ref_meanVLVMAct_r'])
            results_con[motion_type]['video']['scalars']['sim_meanVLVMAct_Asym'] = calc_LSI(results_con[
                motion_type]['video']['scalars']['sim_meanVLVMAct_l'] , results_con[
                motion_type]['video']['scalars']['sim_meanVLVMAct_r'])
            if not fieldStudy:
                results_con[motion_type]['video']['scalars']['so_meanVLVMAct_Asym'] = calc_LSI(results_con[
                    motion_type]['video']['scalars']['so_meanVLVMAct_l'] , results_con[
                    motion_type]['video']['scalars']['so_meanVLVMAct_r'])
                    
            # peak(VL,VM)_l
            idx_VL = [results_con[motion_type]['video']['activations' + actSuffix]['headers'].index('vaslat_l')]
            idx_VM = [results_con[motion_type]['video']['activations' + actSuffix]['headers'].index('vasmed_l')]
            results_con[motion_type]['video']['scalars']['ref_peakVLVMAct_l'] = np.max(np.mean(
                results_con[motion_type]['video']['activations' + actSuffix]['ref'][[idx_VL,idx_VM],:,:],axis=0),axis=1)
            results_con[motion_type]['video']['scalars']['sim_peakVLVMAct_l'] = np.max(np.mean(
                results_con[motion_type]['video']['activations' + actSuffix]['sim'][[idx_VL,idx_VM],:,:],axis=0),axis=1)
            if not fieldStudy:
                results_con[motion_type]['video']['scalars']['so_peakVLVMAct_l'] = np.max(np.mean(
                    results_con[motion_type]['video']['activations']['so'][[idx_VL,idx_VM],:,:],axis=0),axis=1)
            
            # mean(VL,VM)_r
            idx_VL = [results_con[motion_type]['video']['activations' + actSuffix]['headers'].index('vaslat_r')]
            idx_VM = [results_con[motion_type]['video']['activations' + actSuffix]['headers'].index('vasmed_r')]
            results_con[motion_type]['video']['scalars']['ref_peakVLVMAct_r'] = np.max(np.mean(
                results_con[motion_type]['video']['activations' + actSuffix]['ref'][[idx_VL,idx_VM],:,:],axis=0),axis=1)
            results_con[motion_type]['video']['scalars']['sim_peakVLVMAct_r'] = np.max(np.mean(
                results_con[motion_type]['video']['activations' + actSuffix]['sim'][[idx_VL,idx_VM],:,:],axis=0),axis=1)
            if not fieldStudy:
                results_con[motion_type]['video']['scalars']['so_peakVLVMAct_r'] = np.max(np.mean(
                    results_con[motion_type]['video']['activations']['so'][[idx_VL,idx_VM],:,:],axis=0),axis=1)
            
            # mean KEM_both
            results_con[motion_type]['video']['scalars']['ref_peakVLVMAct_both'] = np.mean(np.concatenate((
                results_con[motion_type]['video']['scalars']['ref_peakVLVMAct_r'],
                results_con[motion_type]['video']['scalars']['ref_peakVLVMAct_l'])),axis=0,keepdims=True)
            results_con[motion_type]['video']['scalars']['sim_peakVLVMAct_both'] = np.mean(np.concatenate((
                results_con[motion_type]['video']['scalars']['sim_peakVLVMAct_r'],
                results_con[motion_type]['video']['scalars']['sim_peakVLVMAct_l'])),axis=0,keepdims=True)
            
            # mean(VL,VM) Asym
            results_con[motion_type]['video']['scalars']['ref_peakVLVMAct_Asym'] = calc_LSI(results_con[
                motion_type]['video']['scalars']['ref_peakVLVMAct_l'] , results_con[
                motion_type]['video']['scalars']['ref_peakVLVMAct_r'])
            results_con[motion_type]['video']['scalars']['sim_peakVLVMAct_Asym'] = calc_LSI(results_con[
                motion_type]['video']['scalars']['sim_peakVLVMAct_l'] , results_con[
                motion_type]['video']['scalars']['sim_peakVLVMAct_r'])
            if not fieldStudy:
                results_con[motion_type]['video']['scalars']['so_peakVLVMAct_Asym'] = calc_LSI(results_con[
                    motion_type]['video']['scalars']['so_peakVLVMAct_l'] , results_con[
                    motion_type]['video']['scalars']['so_peakVLVMAct_r'])
                           
            # mean KEM_l
            idx_KEM_l = [results_con[motion_type]['video']['torques_BWht']['headers'].index('knee_angle_l')]
            temp_sim = np.copy(-results_con[motion_type]['video']['torques_BWht']['sim'][idx_KEM_l,:,:])
            # temp_sim[temp_sim<0] = np.nan
            temp_ref = np.copy(-results_con[motion_type]['video']['torques_BWht']['ref'][idx_KEM_l,:,:])
            # temp_ref[temp_sim<0] = np.nan                      
            results_con[motion_type]['video']['scalars']['ref_meanKEM_l'] = np.nanmean(temp_ref,axis=1)
            results_con[motion_type]['video']['scalars']['sim_meanKEM_l'] = np.nanmean(temp_sim,axis=1)
            
            # mean KEM_r
            idx_KEM_r = [results_con[motion_type]['video']['torques_BWht']['headers'].index('knee_angle_r')]
            temp_sim = np.copy(-results_con[motion_type]['video']['torques_BWht']['sim'][idx_KEM_r,:,:])
            # temp_sim[temp_sim<0] = np.nan
            temp_ref = np.copy(-results_con[motion_type]['video']['torques_BWht']['ref'][idx_KEM_r,:,:])
            # temp_ref[temp_sim<0] = np.nan                      
            results_con[motion_type]['video']['scalars']['ref_meanKEM_r'] = np.nanmean(temp_ref,axis=1)
            results_con[motion_type]['video']['scalars']['sim_meanKEM_r'] = np.nanmean(temp_sim,axis=1)
            
            # mean KEM_both
            results_con[motion_type]['video']['scalars']['ref_meanKEM_both'] = np.mean(np.concatenate((
                results_con[motion_type]['video']['scalars']['ref_meanKEM_r'],
                results_con[motion_type]['video']['scalars']['ref_meanKEM_l'])),axis=0,keepdims=True)
            results_con[motion_type]['video']['scalars']['sim_meanKEM_both'] = np.mean(np.concatenate((
                results_con[motion_type]['video']['scalars']['sim_meanKEM_r'],
                results_con[motion_type]['video']['scalars']['sim_meanKEM_l'])),axis=0,keepdims=True)
            
            # mean KEM Asymmetry
            results_con[motion_type]['video']['scalars']['ref_meanKEM_Asym'] = calc_LSI(results_con[
                motion_type]['video']['scalars']['ref_meanKEM_l'] , results_con[
                motion_type]['video']['scalars']['ref_meanKEM_r'])
            results_con[motion_type]['video']['scalars']['sim_meanKEM_Asym'] = calc_LSI(results_con[
                motion_type]['video']['scalars']['sim_meanKEM_l'] , results_con[
                motion_type]['video']['scalars']['sim_meanKEM_r'])
                    
                    
            # peak KEM_l
            idx_KEM_l = [results_con[motion_type]['video']['torques_BWht']['headers'].index('knee_angle_l')]
            results_con[motion_type]['video']['scalars']['ref_peakKEM_l'] = np.max(-
                results_con[motion_type]['video']['torques_BWht']['ref'][idx_KEM_l,:,:],axis=1)
            results_con[motion_type]['video']['scalars']['sim_peakKEM_l'] = np.max(-
                results_con[motion_type]['video']['torques_BWht']['sim'][idx_KEM_l,:,:],axis=1)
            
            # peak KEM_r
            idx_KEM_r = [results_con[motion_type]['video']['torques_BWht']['headers'].index('knee_angle_r')]
            results_con[motion_type]['video']['scalars']['ref_peakKEM_r'] = np.max(-
                results_con[motion_type]['video']['torques_BWht']['ref'][idx_KEM_r,:,:],axis=1)
            results_con[motion_type]['video']['scalars']['sim_peakKEM_r'] = np.max(-
                results_con[motion_type]['video']['torques_BWht']['sim'][idx_KEM_r,:,:],axis=1)
            
            # peak KEM_both
            results_con[motion_type]['video']['scalars']['ref_peakKEM_both'] = np.mean(np.concatenate((
                results_con[motion_type]['video']['scalars']['ref_peakKEM_r'],
                results_con[motion_type]['video']['scalars']['ref_peakKEM_l'])),axis=0,keepdims=True)
            results_con[motion_type]['video']['scalars']['sim_peakKEM_both'] = np.mean(np.concatenate((
                results_con[motion_type]['video']['scalars']['sim_peakKEM_r'],
                results_con[motion_type]['video']['scalars']['sim_peakKEM_l'])),axis=0,keepdims=True)
            
            # peak KEM Asymmetry
            results_con[motion_type]['video']['scalars']['ref_peakKEM_Asym'] = calc_LSI(results_con[
                motion_type]['video']['scalars']['ref_peakKEM_l'] , results_con[
                motion_type]['video']['scalars']['ref_peakKEM_r'])
            results_con[motion_type]['video']['scalars']['sim_peakKEM_Asym'] = calc_LSI(results_con[
                motion_type]['video']['scalars']['sim_peakKEM_l'] , results_con[
                motion_type]['video']['scalars']['sim_peakKEM_r'])
                    
            # mean and KEM Asymmetry - per squat
            if fieldStudy:
                nTrials = len(results_con[motion_type]['video']['scalars']['sim_meanKEM_l'][0])
                results_con[motion_type]['video']['scalars']['sim_meanKEM_dAsym_baseline'] = np.zeros((1,nTrials))
                results_con[motion_type]['video']['scalars']['sim_peakKEM_dAsym_baseline'] = np.zeros((1,nTrials))
                combs = combinations(np.arange(nTrials),2)
                for i,comb in enumerate(combs):
                    results_con[motion_type]['video']['scalars']['sim_meanKEM_dAsym_baseline'][0,i] =  results_con[
                        motion_type]['video']['scalars']['sim_meanKEM_Asym'][-1][comb[0]] - results_con[
                        motion_type]['video']['scalars']['sim_meanKEM_Asym'][-1][comb[1]]
                    results_con[motion_type]['video']['scalars']['sim_peakKEM_dAsym_baseline'][0,i] =  results_con[
                        motion_type]['video']['scalars']['sim_peakKEM_Asym'][-1][comb[0]] - results_con[
                        motion_type]['video']['scalars']['sim_peakKEM_Asym'][-1][comb[1]]
                    
        elif 'STS' in motion_type:
            actSuffix = '' 
            # mean(VL,VM)_l
            idx_VL = [results_con[motion_type]['video']['activations' + actSuffix]['headers'].index('vaslat_l')]
            idx_VM = [results_con[motion_type]['video']['activations' + actSuffix]['headers'].index('vasmed_l')]
            results_con[motion_type]['video']['scalars']['ref_meanVLVMAct_l'] = np.mean(np.mean(
                results_con[motion_type]['video']['activations' + actSuffix]['ref'][[idx_VL,idx_VM],:,:],axis=0),axis=1)
            results_con[motion_type]['video']['scalars']['sim_meanVLVMAct_l'] = np.mean(np.mean(
                results_con[motion_type]['video']['activations' + actSuffix]['sim'][[idx_VL,idx_VM],:,:],axis=0),axis=1)

            
            # mean(VL,VM)_r
            idx_VL = [results_con[motion_type]['video']['activations' + actSuffix]['headers'].index('vaslat_r')]
            idx_VM = [results_con[motion_type]['video']['activations' + actSuffix]['headers'].index('vasmed_r')]
            results_con[motion_type]['video']['scalars']['ref_meanVLVMAct_r'] = np.mean(np.mean(
                results_con[motion_type]['video']['activations' + actSuffix]['ref'][[idx_VL,idx_VM],:,:],axis=0),axis=1)
            results_con[motion_type]['video']['scalars']['sim_meanVLVMAct_r'] = np.mean(np.mean(
                results_con[motion_type]['video']['activations' + actSuffix]['sim'][[idx_VL,idx_VM],:,:],axis=0),axis=1)

            
            # mean act_both
            results_con[motion_type]['video']['scalars']['ref_meanVLVMAct_both'] = np.mean(np.concatenate((
                results_con[motion_type]['video']['scalars']['ref_meanVLVMAct_r'],
                results_con[motion_type]['video']['scalars']['ref_meanVLVMAct_l'])),axis=0,keepdims=True)
            results_con[motion_type]['video']['scalars']['sim_meanVLVMAct_both'] = np.mean(np.concatenate((
                results_con[motion_type]['video']['scalars']['sim_meanVLVMAct_r'],
                results_con[motion_type]['video']['scalars']['sim_meanVLVMAct_l'])),axis=0,keepdims=True)
            
            # mean(VL,VM) Asym
            results_con[motion_type]['video']['scalars']['ref_meanVLVMAct_Asym'] = calc_LSI(results_con[
                motion_type]['video']['scalars']['ref_meanVLVMAct_l'] , results_con[
                motion_type]['video']['scalars']['ref_meanVLVMAct_r'])
            results_con[motion_type]['video']['scalars']['sim_meanVLVMAct_Asym'] = calc_LSI(results_con[
                motion_type]['video']['scalars']['sim_meanVLVMAct_l'] , results_con[
                motion_type]['video']['scalars']['sim_meanVLVMAct_r'])
                    
            # peak(VL,VM)_l
            idx_VL = [results_con[motion_type]['video']['activations' + actSuffix]['headers'].index('vaslat_l')]
            idx_VM = [results_con[motion_type]['video']['activations' + actSuffix]['headers'].index('vasmed_l')]
            results_con[motion_type]['video']['scalars']['ref_peakVLVMAct_l'] = np.max(np.mean(
                results_con[motion_type]['video']['activations' + actSuffix]['ref'][[idx_VL,idx_VM],:,:],axis=0),axis=1)
            results_con[motion_type]['video']['scalars']['sim_peakVLVMAct_l'] = np.max(np.mean(
                results_con[motion_type]['video']['activations' + actSuffix]['sim'][[idx_VL,idx_VM],:,:],axis=0),axis=1)

            
            # mean(VL,VM)_r
            idx_VL = [results_con[motion_type]['video']['activations' + actSuffix]['headers'].index('vaslat_r')]
            idx_VM = [results_con[motion_type]['video']['activations' + actSuffix]['headers'].index('vasmed_r')]
            results_con[motion_type]['video']['scalars']['ref_peakVLVMAct_r'] = np.max(np.mean(
                results_con[motion_type]['video']['activations' + actSuffix]['ref'][[idx_VL,idx_VM],:,:],axis=0),axis=1)
            results_con[motion_type]['video']['scalars']['sim_peakVLVMAct_r'] = np.max(np.mean(
                results_con[motion_type]['video']['activations' + actSuffix]['sim'][[idx_VL,idx_VM],:,:],axis=0),axis=1)

            
            # peak act_both
            results_con[motion_type]['video']['scalars']['ref_peakVLVMAct_both'] = np.mean(np.concatenate((
                results_con[motion_type]['video']['scalars']['ref_peakVLVMAct_r'],
                results_con[motion_type]['video']['scalars']['ref_peakVLVMAct_l'])),axis=0,keepdims=True)
            results_con[motion_type]['video']['scalars']['sim_peakVLVMAct_both'] = np.mean(np.concatenate((
                results_con[motion_type]['video']['scalars']['sim_peakVLVMAct_r'],
                results_con[motion_type]['video']['scalars']['sim_peakVLVMAct_l'])),axis=0,keepdims=True)
            
            # mean(VL,VM) Asym
            results_con[motion_type]['video']['scalars']['ref_peakVLVMAct_Asym'] = calc_LSI(results_con[
                motion_type]['video']['scalars']['ref_peakVLVMAct_l'] , results_con[
                motion_type]['video']['scalars']['ref_peakVLVMAct_r'])
            results_con[motion_type]['video']['scalars']['sim_peakVLVMAct_Asym'] = calc_LSI(results_con[
                motion_type]['video']['scalars']['sim_peakVLVMAct_l'] , results_con[
                motion_type]['video']['scalars']['sim_peakVLVMAct_r'])
            
            # mean KEM_l
            idx_KEM_l = [results_con[motion_type]['video']['torques_BWht']['headers'].index('knee_angle_l')]
            temp_sim = np.copy(-results_con[motion_type]['video']['torques_BWht']['sim'][idx_KEM_l,:,:])
            # temp_sim[temp_sim<0] = np.nan
            temp_ref = np.copy(-results_con[motion_type]['video']['torques_BWht']['ref'][idx_KEM_l,:,:])
            # temp_ref[temp_sim<0] = np.nan                      
            results_con[motion_type]['video']['scalars']['ref_meanKEM_l'] = np.nanmean(temp_ref,axis=1)
            results_con[motion_type]['video']['scalars']['sim_meanKEM_l'] = np.nanmean(temp_sim,axis=1)
            
            # mean KEM_r
            idx_KEM_r = [results_con[motion_type]['video']['torques_BWht']['headers'].index('knee_angle_r')]
            temp_sim = np.copy(-results_con[motion_type]['video']['torques_BWht']['sim'][idx_KEM_r,:,:])
            # temp_sim[temp_sim<0] = np.nan
            temp_ref = np.copy(-results_con[motion_type]['video']['torques_BWht']['ref'][idx_KEM_r,:,:])
            # temp_ref[temp_sim<0] = np.nan                      
            results_con[motion_type]['video']['scalars']['ref_meanKEM_r'] = np.nanmean(temp_ref,axis=1)
            results_con[motion_type]['video']['scalars']['sim_meanKEM_r'] = np.nanmean(temp_sim,axis=1)
            
            # mean KEM_both
            results_con[motion_type]['video']['scalars']['ref_meanKEM_both'] = np.mean(np.concatenate((
                results_con[motion_type]['video']['scalars']['ref_meanKEM_r'],
                results_con[motion_type]['video']['scalars']['ref_meanKEM_l'])),axis=0,keepdims=True)
            results_con[motion_type]['video']['scalars']['sim_meanKEM_both'] = np.mean(np.concatenate((
                results_con[motion_type]['video']['scalars']['sim_meanKEM_r'],
                results_con[motion_type]['video']['scalars']['sim_meanKEM_l'])),axis=0,keepdims=True)
            
            # mean HEM_l
            idx_HEM_l = [results_con[motion_type]['video']['torques_BWht']['headers'].index('hip_flexion_l')]
            temp_sim = np.copy(-results_con[motion_type]['video']['torques_BWht']['sim'][idx_HEM_l,:,:])
            # temp_sim[temp_sim<0] = np.nan
            temp_ref = np.copy(-results_con[motion_type]['video']['torques_BWht']['ref'][idx_HEM_l,:,:])
            # temp_ref[temp_sim<0] = np.nan                      
            results_con[motion_type]['video']['scalars']['ref_meanHEM_l'] = np.nanmean(temp_ref,axis=1)
            results_con[motion_type]['video']['scalars']['sim_meanHEM_l'] = np.nanmean(temp_sim,axis=1)
            
            # mean HEM_r
            idx_HEM_r = [results_con[motion_type]['video']['torques_BWht']['headers'].index('hip_flexion_r')]
            temp_sim = np.copy(-results_con[motion_type]['video']['torques_BWht']['sim'][idx_HEM_r,:,:])
            # temp_sim[temp_sim<0] = np.nan
            temp_ref = np.copy(-results_con[motion_type]['video']['torques_BWht']['ref'][idx_HEM_r,:,:])
            # temp_ref[temp_sim<0] = np.nan                      
            results_con[motion_type]['video']['scalars']['ref_meanHEM_r'] = np.nanmean(temp_ref,axis=1)
            results_con[motion_type]['video']['scalars']['sim_meanHEM_r'] = np.nanmean(temp_sim,axis=1)
            
            # mean HEM_both
            results_con[motion_type]['video']['scalars']['ref_meanHEM_both'] = np.mean(np.concatenate((
                results_con[motion_type]['video']['scalars']['ref_meanHEM_r'],
                results_con[motion_type]['video']['scalars']['ref_meanHEM_l'])),axis=0,keepdims=True)
            results_con[motion_type]['video']['scalars']['sim_meanHEM_both'] = np.mean(np.concatenate((
                results_con[motion_type]['video']['scalars']['sim_meanHEM_r'],
                results_con[motion_type]['video']['scalars']['sim_meanHEM_l'])),axis=0,keepdims=True)
            
            # mean APM_l
            idx_APM_l = [results_con[motion_type]['video']['torques_BWht']['headers'].index('ankle_angle_l')]
            temp_sim = np.copy(-results_con[motion_type]['video']['torques_BWht']['sim'][idx_APM_l,:,:])
            # temp_sim[temp_sim<0] = np.nan
            temp_ref = np.copy(-results_con[motion_type]['video']['torques_BWht']['ref'][idx_APM_l,:,:])
            # temp_ref[temp_sim<0] = np.nan                      
            results_con[motion_type]['video']['scalars']['ref_meanAPM_l'] = np.nanmean(temp_ref,axis=1)
            results_con[motion_type]['video']['scalars']['sim_meanAPM_l'] = np.nanmean(temp_sim,axis=1)
            
            # mean APM_r
            idx_APM_r = [results_con[motion_type]['video']['torques_BWht']['headers'].index('ankle_angle_r')]
            temp_sim = np.copy(-results_con[motion_type]['video']['torques_BWht']['sim'][idx_APM_r,:,:])
            # temp_sim[temp_sim<0] = np.nan
            temp_ref = np.copy(-results_con[motion_type]['video']['torques_BWht']['ref'][idx_APM_r,:,:])
            # temp_ref[temp_sim<0] = np.nan                      
            results_con[motion_type]['video']['scalars']['ref_meanAPM_r'] = np.nanmean(temp_ref,axis=1)
            results_con[motion_type]['video']['scalars']['sim_meanAPM_r'] = np.nanmean(temp_sim,axis=1)
            
            # mean APM_both
            results_con[motion_type]['video']['scalars']['ref_meanAPM_both'] = np.mean(np.concatenate((
                results_con[motion_type]['video']['scalars']['ref_meanAPM_r'],
                results_con[motion_type]['video']['scalars']['ref_meanAPM_l'])),axis=0,keepdims=True)
            results_con[motion_type]['video']['scalars']['sim_meanAPM_both'] = np.mean(np.concatenate((
                results_con[motion_type]['video']['scalars']['sim_meanAPM_r'],
                results_con[motion_type]['video']['scalars']['sim_meanAPM_l'])),axis=0,keepdims=True)
            
            # mean KEM Asymmetry
            results_con[motion_type]['video']['scalars']['ref_meanKEM_Asym'] = calc_LSI(results_con[
                motion_type]['video']['scalars']['ref_meanKEM_l'] , results_con[
                motion_type]['video']['scalars']['ref_meanKEM_r'])
            results_con[motion_type]['video']['scalars']['sim_meanKEM_Asym'] = calc_LSI(results_con[
                motion_type]['video']['scalars']['sim_meanKEM_l'] , results_con[
                motion_type]['video']['scalars']['sim_meanKEM_r'])
                    
            # peak KEM_l
            idx_KEM_l = [results_con[motion_type]['video']['torques_BWht']['headers'].index('knee_angle_l')]
            results_con[motion_type]['video']['scalars']['ref_peakKEM_l'] = np.max(-
                results_con[motion_type]['video']['torques_BWht']['ref'][idx_KEM_l,:,:],axis=1)
            results_con[motion_type]['video']['scalars']['sim_peakKEM_l'] = np.max(-
                results_con[motion_type]['video']['torques_BWht']['sim'][idx_KEM_l,:,:],axis=1)
            
            # peak KEM_r
            idx_KEM_r = [results_con[motion_type]['video']['torques_BWht']['headers'].index('knee_angle_r')]
            results_con[motion_type]['video']['scalars']['ref_peakKEM_r'] = np.max(-
                results_con[motion_type]['video']['torques_BWht']['ref'][idx_KEM_r,:,:],axis=1)
            results_con[motion_type]['video']['scalars']['sim_peakKEM_r'] = np.max(-
                results_con[motion_type]['video']['torques_BWht']['sim'][idx_KEM_r,:,:],axis=1)
            
            # peak KEM_both
            results_con[motion_type]['video']['scalars']['ref_peakKEM_both'] = np.mean(np.concatenate((
                results_con[motion_type]['video']['scalars']['ref_peakKEM_r'],
                results_con[motion_type]['video']['scalars']['ref_peakKEM_l'])),axis=0,keepdims=True)
            results_con[motion_type]['video']['scalars']['sim_peakKEM_both'] = np.mean(np.concatenate((
                results_con[motion_type]['video']['scalars']['sim_peakKEM_r'],
                results_con[motion_type]['video']['scalars']['sim_peakKEM_l'])),axis=0,keepdims=True)
            
            # peak KEM Asymmetry
            results_con[motion_type]['video']['scalars']['ref_peakKEM_Asym'] = calc_LSI(results_con[
                motion_type]['video']['scalars']['ref_peakKEM_l'] , results_con[
                motion_type]['video']['scalars']['ref_peakKEM_r'])
            results_con[motion_type]['video']['scalars']['sim_peakKEM_Asym'] = calc_LSI(results_con[
                motion_type]['video']['scalars']['sim_peakKEM_l'] , results_con[
                motion_type]['video']['scalars']['sim_peakKEM_r'])
                    
        else:
            raise Exception('invalid motion type in scalar computation:' + motion_type)
            
        #%% Take means across trials
        results_con[motion_type]['video']['scalar_means'] = {}
        for scalar in results_con[motion_type]['video']['scalars']:
            results_con[motion_type]['video']['scalar_means'][scalar + '_mean'] = np.mean(
                results_con[motion_type]['video']['scalars'][scalar])
        
        #%% Compute RMSEs, MAEs, MAE as % range
        nCases = len(cases)
        for variable_type in results:
            if variable_type == 'GRFs_BW_peaks' or variable_type == 'GRFs_peaks':
                continue
            nHeaders = len(results_con[motion_type]['video'][variable_type]['headers']) 
            results_con[motion_type]['video'][variable_type]['rmse'] = np.zeros((nHeaders,1,nCases))
            results_con[motion_type]['video'][variable_type]['mae'] = np.zeros((nHeaders,1,nCases))
            results_con[motion_type]['video'][variable_type]['mape'] = np.zeros((nHeaders,1,nCases))
            for iHeader in range(nHeaders):
                for iCase in range(nCases): 
                   y_true = results_con[motion_type]['video'][variable_type]['ref'][iHeader,:,iCase]
                   y_pred = results_con[motion_type]['video'][variable_type]['sim'][iHeader,:,iCase]
                   if not any(np.isnan(np.concatenate((y_true,y_pred)))):
                       results_con[motion_type]['video'][variable_type]['rmse'][iHeader,0,iCase] = mean_squared_error(
                           y_true,y_pred,squared=False)
                       results_con[motion_type]['video'][variable_type]['mae'][iHeader,0,iCase] = mean_absolute_error(
                           y_true,y_pred)
                       results_con[motion_type]['video'][variable_type]['mape'][iHeader,0,iCase] = results_con[
                           motion_type]['video'][variable_type]['mae'][iHeader,0,iCase] / np.ptp(y_true) * 100
                   else:
                       # there are nans for activations that weren't measured
                       results_con[motion_type]['video'][variable_type]['rmse'][iHeader,0,iCase] = np.nan
                       results_con[motion_type]['video'][variable_type]['mae'][iHeader,0,iCase] = np.nan
                       results_con[motion_type]['video'][variable_type]['mape'][iHeader,0,iCase] = np.nan

                # Mean rmse and mae
                results_con[motion_type]['video'][variable_type]['rmse_mean'] = np.mean(
                   results_con[motion_type]['video'][variable_type]['rmse'],axis=2)
                results_con[motion_type]['video'][variable_type]['mae_mean'] = np.mean(
                   results_con[motion_type]['video'][variable_type]['mae'],axis=2)    
                results_con[motion_type]['video'][variable_type]['mape_mean'] = np.mean(
                   results_con[motion_type]['video'][variable_type]['mape'],axis=2)   
                  
        # %% Save concatenated results with scalars on a per-subject basis for loading and plotting later
        if saveResults:
            bD_suff = ''
            if fieldStudy:
                resultPath = os.path.join(pathOSData,'allActivityResults.npy')
                try:
                    os.remove(os.path.join(pathOSData,'allActivityResults_fieldStudy.npy'))
                    print('removed')
                except:
                    test=1
            else:
                resultPath = os.path.join(pathOSData,'allActivityResults.npy')
                  
            if not os.path.exists(resultPath): 
                results_all = {}
            else:  
                results_all = np.load(resultPath,allow_pickle=True).item()
            if not motion_type in results_all:
                results_all[motion_type] = {}
            results_all[motion_type] = results_con[motion_type]
            np.save(resultPath, results_all)
        