# -*- coding: utf-8 -*-
"""
Created on Wed May  4 21:57:35 2022

@author: suhlr
"""
import utilsAuth
import opensim
import utils
import os
import glob

import numpy as np
import scipy.interpolate as interpolate


def downloadKinematics(session_id,folder=None,trialNames=None):
    # Login to access opencap data from server. 
    utilsAuth.getToken()
    
    if folder is None:
        folder = os.getcwd()
    
    os.makedirs(folder,exist_ok=True)
    
    sessionJson = utils.getSessionJson(session_id)
    
    # Get scaled model
    neutral_id = utils.getNeutralTrialID(session_id)
    utils.getMotionData(neutral_id,folder,simplePath=True)
    utils.getModelAndMetadata(session_id,folder,simplePath =True)
    
    # get session trial names
    sessionTrialNames = [t['name'] for t in sessionJson['trials']]
    
    # Check if trialnames they asked for are in sessionTrialNames
    if trialNames != None:
        [print(t + ' not in session trial names.') for t in trialNames if t not in sessionTrialNames]
    
    loadedTrialNames = []
    for trialDict in sessionJson['trials']:
        if trialNames is not None and trialDict['name'] not in trialNames:
            continue
        trial_id = trialDict['id']
        utils.getMotionData(trial_id,folder,simplePath=True)
        loadedTrialNames.append(trialDict['name'])
        
    return loadedTrialNames
  

def calcCenterOfMassTrajectory(folder,trialNames=None,coordinateFilterFreq=-1,COMFilterFreq=-1):
    
    modelPath = glob.glob(os.path.join(folder,'OpenSimData','Model','*.osim'))[0]
    kinematicPaths = glob.glob(os.path.join(folder,'OpenSimData','Kinematics','*.mot'))
    modProc = opensim.ModelProcessor(os.path.abspath(modelPath))
    modProc.append(opensim.ModOpRemoveMuscles())
    model = modProc.process()
    state = model.initSystem()
    
    kinematicsCOM = []
    
    for kinematicPath in kinematicPaths:
        kinematicPath = os.path.abspath(kinematicPath)
        kinematicRoot,fileName = os.path.split(kinematicPath)
        trialName,_ = os.path.splitext(fileName)
        if trialNames is not None:
            if trialName not in trialNames:
                continue
        
        # get states from motion
        statesTable = writeStatesFromMotion(kinematicPath,model,filtFreq=coordinateFilterFreq)
        
        # Load states trajectory
        tableProc = opensim.TableProcessor(statesTable)
        statesTable = tableProc.processAndConvertToRadians(model)
        statesTraj = opensim.StatesTrajectory.createFromStatesTable(model, statesTable)
        
        # initialize output dict
        nSteps = statesTraj.getSize()
        inputDict = {'name':trialName}
        inputDict['fields'] = ['time']
        params = ['pos','vel','acc']
        dirs = ['x','y','z']
        [inputDict['fields'].append(p+'_'+d) for p in params for d in dirs]
        inputDict['data'] = np.ndarray((nSteps,len(inputDict['fields'])))     
        
        # Compute COM position and velocity
        for iTime in range(statesTraj.getSize()):
            state = statesTraj[iTime]
            model.realizeAcceleration(state)
            inputDict['data'][iTime,0] = state.getTime()
            inputDict['data'][iTime,1:4] = model.calcMassCenterPosition(state).to_numpy()
            inputDict['data'][iTime,4:7] = model.calcMassCenterVelocity(state).to_numpy()
        
        # dSpline/dt differences for accelerations because opensim calculation
        # realizes to dynamics, so you need ground forces
        for i in range(4,7,1):
            spline = interpolate.InterpolatedUnivariateSpline(
                        inputDict['data'][:,0], 
                        inputDict['data'][:,i], k=3)
            inputDict['data'][:,i] = spline(
                inputDict['data'][:,0])
            splineD1 = spline.derivative(n=1)
            inputDict['data'][:,i+3] = splineD1(
                inputDict['data'][:,0])
       
        
        # Filter the positions, velocities and accelerations
        if COMFilterFreq >0:
            inputDict['data'] = utils.lowpassFilter(inputDict['data'],
                                                    filtFreq=COMFilterFreq)
            
        kinematicsCOM.append(inputDict)
    
    return kinematicsCOM

def writeStatesFromMotion(filePath,model,filtFreq=-1):
        kinematicPath = os.path.abspath(filePath)
        kinematicRoot,fileName = os.path.split(kinematicPath)
        trialName,_ = os.path.splitext(fileName)
        
        # create states from motion - this is a hack to use Moco conveniences
        track = opensim.MocoTrack()
        track.setName(trialName)
        modProc = opensim.ModelProcessor(model)
        modProc.append(opensim.ModOpRemoveMuscles())
        track.setModel(modProc)
        table = opensim.TimeSeriesTable(kinematicPath)
        time = table.getIndependentColumn()
        t_initial = time[0]
        t_final = time[-1]
        tabProc = opensim.TableProcessor(table)
        tabProc.append(opensim.TabOpLowPassFilter(filtFreq))
        tabProc.append(opensim.TabOpUseAbsoluteStateNames())
        track.setStatesReference(tabProc)
        track.set_track_reference_position_derivatives(True)
        track.initialize()
        
        # move and rename file
        oldFilePath = trialName + '_tracked_states.sto'
        newFilePath = os.path.join(kinematicRoot,trialName + '_states.sto')
        if filtFreq < 0:
            table = opensim.TimeSeriesTable(newFilePath)
        else:
            table = opensim.TimeSeriesTable(oldFilePath)
            table.trim(t_initial,t_final)
            opensim.STOFileAdapter().write(table,newFilePath)
            os.remove(oldFilePath)
 
        return table

