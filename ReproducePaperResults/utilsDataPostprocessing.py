# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 15:11:22 2021

@author: suhlr
"""

import sys
sys.path.append("../../") # utilities in child directory
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.interpolate import interp1d



def segmentSquats(ikFilePath,pelvis_ty=None, timeVec=None, visualize=False,
                  filter_pelvis_ty=False, cutoff_frequency=6, height=.2):
    
    # Load from file if the vectors aren't supplied directly
    if pelvis_ty is None and timeVec is None:
        ikResults = storage2df(ikFilePath,headers={'pelvis_ty'})
        timeVec = ikResults['time']
        if filter_pelvis_ty:
            from variousFunctions import filterNumpyArray
            pelvis_ty = filterNumpyArray(
                ikResults['pelvis_ty'].to_numpy(), timeVec.to_numpy(), 
                cutoff_frequency=cutoff_frequency)
        else:
            pelvis_ty = ikResults['pelvis_ty']
    
    dt = timeVec[1] - timeVec[0]

    # identify the minimum points
    pelvSignal = np.array(-pelvis_ty - np.min(-pelvis_ty))
    pelvSignalPos = np.array(pelvis_ty - np.min(pelvis_ty))
    idxMinPelvTy,_ = signal.find_peaks(pelvSignal,distance=.7/dt,height=height)
    
    # find the max adjacent to all of the mins
    minIdxOld = 0
    startFinishInds = []
    for i, minIdx in enumerate(idxMinPelvTy):
        if i<len(idxMinPelvTy)-1:
            nextIdx = idxMinPelvTy[i+1]
        else:
            nextIdx = len(pelvSignalPos)
        startIdx = np.argmax(pelvSignalPos[minIdxOld:minIdx]) + minIdxOld
        endIdx = np.argmax(pelvSignalPos[minIdx:nextIdx]) + minIdx
        startFinishInds.append([startIdx,endIdx])
        minIdxOld = np.copy(minIdx)
        
    startFinishTimes = [timeVec[i].tolist() for i in startFinishInds]
    
    if visualize:
        plt.figure()     
        plt.plot(pelvSignal)
        for val in startFinishInds:
            plt.plot(val,pelvSignal[val],marker='o',markerfacecolor='k',markeredgecolor='none',linestyle='none')
        plt.draw()
    
    return startFinishInds, startFinishTimes

def segmentSTS(ikFilePath,pelvis_ty=None,timeVec=None,velSeated=0.3,velStanding=0.3,
               visualize=False, cutoff_frequency=None, delay=0):
    
    # Load from file if the vectors aren't supplied directly
    if pelvis_ty is None and timeVec is None:
        ikResults = storage2df(ikFilePath,headers={'pelvis_ty'})
        if cutoff_frequency:
            ikResults = filterDataFrame(ikResults,
                                        cutoff_frequency=cutoff_frequency)            
        timeVec = ikResults['time']
        pelvis_ty = ikResults['pelvis_ty']
    
    dt = timeVec[1] - timeVec[0]
    

    # identify the minimum points
    pelvSignal = np.array(pelvis_ty - np.min(pelvis_ty))
    pelvVel = np.diff(pelvSignal,append=0) / dt
    # pelvSignalNeg = np.array(pelvis_ty - np.min(pelvis_ty))
    idxMaxPelvTy,_ = signal.find_peaks(pelvSignal,distance=.9/dt,height=.2,prominence=.2)
    
    # find the max adjacent to all of the mins
    maxIdxOld = 0
    startFinishInds = []
    for i, maxIdx in enumerate(idxMaxPelvTy):     
        # Find velocity peak to left of pelv_ty peak
        vels = pelvVel[maxIdxOld:maxIdx]
        velPeak,peakVals = signal.find_peaks(vels,distance=.9/dt,height=.2) 
        velPeak = velPeak[np.argmax(peakVals['peak_heights'])] + maxIdxOld
        
        velsLeftOfPeak = np.flip(pelvVel[maxIdxOld:velPeak])
        velsRightOfPeak = pelvVel[velPeak:]
        
        # trace left off the pelv_ty peak and find first index where velocity<velSeated m/s
        slowingIndLeft = np.argwhere(velsLeftOfPeak<velSeated)[0]
        startIdx = velPeak - slowingIndLeft
        slowingIndRight = np.argwhere(velsRightOfPeak<velStanding)[0]
        endIdx = velPeak + slowingIndRight
        startFinishInds.append([startIdx[0],endIdx[0]])
        maxIdxOld = np.copy(maxIdx)
    
    sf = 1/np.round(np.mean(np.round(timeVec.to_numpy()[1:] - timeVec.to_numpy()[:-1],2)),16)
    # delete just for gut check
    if visualize:        
        plt.figure()     
        plt.plot(pelvSignal)
        for val in startFinishInds:
            plt.plot(val,pelvSignal[val],marker='o',markerFacecolor='k',markeredgecolor='none',linestyle='none')
            
            val2 = val[0] + int(delay*sf)
            if val2 > 0:
                plt.plot(val2,pelvSignal[val2],marker='o',markerFacecolor='r',markeredgecolor='none',linestyle='none')
        plt.show()
        
    startFinishTimes = [timeVec[i].tolist() for i in startFinishInds]
    
    startFinishIndsDelay = []
    for i in startFinishInds:
        c_i = []
        for c_j, j in enumerate(i):
            if c_j == 0:
                c_i.append(j + int(delay*sf))
            else:
                c_i.append(j)
        startFinishIndsDelay.append(c_i)
    startFinishTimesDelay = [timeVec[i].tolist() for i in startFinishIndsDelay]
    
    return startFinishInds, startFinishTimes, startFinishIndsDelay, startFinishTimesDelay

def segmentDJ(pathForceFile):
    headers_force = [
    'R_ground_force_vx', 'R_ground_force_vy', 'R_ground_force_vz', 
    'R_ground_force_px', 'R_ground_force_py', 'R_ground_force_pz',
    'R_ground_torque_x', 'R_ground_torque_y', 'R_ground_torque_z',
    'L_ground_force_vx', 'L_ground_force_vy', 'L_ground_force_vz',
    'L_ground_force_px', 'L_ground_force_py', 'L_ground_force_pz',
    'L_ground_torque_x', 'L_ground_torque_y', 'L_ground_torque_z']
    forceData = storage2df(pathForceFile, headers_force)
    # Select all vertical force vectors.
    verticalForces = np.concatenate(( 
        np.expand_dims(forceData['R_ground_force_vy'].to_numpy(), axis=1),
        np.expand_dims(forceData['L_ground_force_vy'].to_numpy(), axis=1)), axis=1)
    sumVerticalForces = np.sum(verticalForces, axis=1)
    diffForces = np.diff(sumVerticalForces)
    startIdx = np.argwhere(diffForces)[0][0] + 1
    endIdx = np.argwhere(diffForces[startIdx-1:] == 0)[0][0] + startIdx - 2
    startTime = np.round(forceData.iloc[startIdx]['time'], 2)
    endTime = np.round(forceData.iloc[endIdx]['time'], 2)     
    
    startFinishInds = [startIdx, endIdx]
    startFinishTimes = [startTime, endTime]           

    return startFinishInds, startFinishTimes

def segmentWalkStance(pathForceFile,vertForceHeader=['L_ground_force_vy'],forceThreshold=10):
    forceData = storage2df(pathForceFile, vertForceHeader)
    verticalForce = np.expand_dims(forceData[vertForceHeader[0]].to_numpy(), axis=1)
    startIdx = np.argwhere(verticalForce>forceThreshold)[0][0]
    endIdx = np.argwhere(verticalForce>forceThreshold)[-1][0]
    startTime = forceData['time'][startIdx]
    endTime = forceData['time'][endIdx]
    startFinishInds = [startIdx, endIdx]
    startFinishTimes = [startTime, endTime]
            
    return startFinishInds, startFinishTimes

def getIndsFromTimes(times,timeVec):
    indsOut = [np.argmin(np.abs(timeVec-t)) for t in times]
    return indsOut

def calc_r2(y_true,y_pred):
    r2 = np.square(np.corrcoef(y_true,y_pred))[0,1]
    return r2

def calc_LSI(injured,uninjured):
    LSI = 1-(injured-uninjured)/uninjured
    # LSI = injured/uninjured
    return LSI

def filterDataFrame(dataFrame, cutoff_frequency=6, order=4):
    
    # Filter data    
    fs = 1 / np.round(np.mean(np.diff(dataFrame['time'])), 16)
    fc = cutoff_frequency # Cut-off frequency of the filter
    w = fc / (fs / 2) # Normalize the frequency
    b, a = signal.butter(order/2, w, 'low')  
    output = signal.filtfilt(
        b, a, dataFrame.loc[:, dataFrame.columns != 'time'], axis=0, 
        padtype='odd', padlen=3*(max(len(b),len(a))-1))
    columns_keys = [i for i in dataFrame.columns if i != 'time']
    output = pd.DataFrame(data=output, columns=columns_keys)
    dataFrameFilt = pd.concat(
        [pd.DataFrame(data=dataFrame['time'].to_numpy(), columns=['time']), 
         output], axis=1)
    
    print('dataFrame filtered at {}Hz.'.format(cutoff_frequency))    
    
    return dataFrameFilt

def interpolateNumpyArray(data, N):         
    
    time = data[0, :]
    tOut = np.linspace(time[0], time[-1], N)
    dataInterp = np.zeros([data.shape[0], N])    
    dataInterp[0, :] = tOut
    for i in range(data.shape[0]-1):
        set_interp = interp1d(time, data[i+1, :])
        dataInterp[i+1,:] = set_interp(tOut)
        
    return dataInterp

def storage2numpy(storage_file, excess_header_entries=0):
    """Returns the data from a storage file in a numpy format. Skips all lines
    up to and including the line that says 'endheader'.
    Parameters
    ----------
    storage_file : str
        Path to an OpenSim Storage (.sto) file.
    Returns
    -------
    data : np.ndarray (or numpy structure array or something?)
        Contains all columns from the storage file, indexable by column name.
    excess_header_entries : int, optional
        If the header row has more names in it than there are data columns.
        We'll ignore this many header row entries from the end of the header
        row. This argument allows for a hacky fix to an issue that arises from
        Static Optimization '.sto' outputs.
    Examples
    --------
    Columns from the storage file can be obtained as follows:
        >>> data = storage2numpy('<filename>')
        >>> data['ground_force_vy']
    """
    # What's the line number of the line containing 'endheader'?
    f = open(storage_file, 'r')

    header_line = False
    for i, line in enumerate(f):
        if header_line:
            column_names = line.split()
            break
        if line.count('endheader') != 0:
            line_number_of_line_containing_endheader = i + 1
            header_line = True
    f.close()

    # With this information, go get the data.
    if excess_header_entries == 0:
        names = True
        skip_header = line_number_of_line_containing_endheader
    else:
        names = column_names[:-excess_header_entries]
        skip_header = line_number_of_line_containing_endheader + 1
    data = np.genfromtxt(storage_file, names=names,
            skip_header=skip_header)

    return data

def storage2df(storage_file, headers):
    # Extract data
    data = storage2numpy(storage_file)
    out = pd.DataFrame(data=data['time'], columns=['time'])    
    for count, header in enumerate(headers):
        out.insert(count + 1, header, data[header])    
    
    return out