"""
Created on Tue Dec 10 13:53:43 2024

@author: Matt
"""

import sys
import os
sys.path.append(os.path.abspath('./..'))
import numpy as np
import pandas as pd
import traceback
import opensim as osim
import matplotlib.pyplot as plt

import logging
logging.basicConfig(level=logging.INFO)


from utilsAugmenter import augmentTRC
from utils import importMetadata
from utilsOpenSim import runIKTool, generateVisualizerJson
from utilsDataman import TRCFile

# Session ID to download
sessionName = 'OpenCapData_b39b10d1-17c7-4976-b06c-a6aaf33fead2'
initialTime = 4.3
finalTime = 5.3
trialName = 'gait_5'

# Stuff that needs to be defined but may need to adjust
augmenter_model = 'v0.3'
offset = True

# Local path to download sessions to
#dataDir = os.path.abspath(os.path.join(os.path.dirname(__file__),'Data'))
## FOR DEBUG
baseDir = os.getcwd()
dataDir = os.path.abspath(os.path.join(baseDir,'Data'))

# Name downloaded folder with subject name from web app (True), or the session_id (False)
useSubjectIdentifierAsFolderName = False 
rotateData = True
runOpenSimPipeline = True
overwriteAugmenterModel = True

# %% Paths and metadata. This gets defined through web app.
baseDir = os.path.dirname(os.path.abspath(__file__))

if 'dataDir' not in locals():
    sessionDir = os.path.join(baseDir, 'Data', sessionName)
else:
    sessionDir = os.path.join(dataDir, sessionName)
sessionMetadata = importMetadata(os.path.join(sessionDir,
                                              'sessionMetadata.yaml'))

# If augmenter model defined through web app.
# If overwriteAugmenterModel is True, the augmenter model is the one
# passed as an argument to main(). This is useful for local testing.
if 'augmentermodel' in sessionMetadata and not overwriteAugmenterModel:
    augmenterModel = sessionMetadata['augmentermodel']
else:
    augmenterModel = augmenter_model

# Define where the marker data lives
markerDataFolderName = os.path.join(sessionDir,'MarkerData')
markerDataPath = os.path.join(markerDataFolderName,trialName + ".trc")

#Create folder to put trimmed data into
trimmedPath = os.path.join(markerDataFolderName,"trimmedTRCs")
os.makedirs(trimmedPath, exist_ok=True)

# Set output file name.
pathOutputFiles = {}
pathOutputFiles[trialName] = os.path.join(trimmedPath,
                                              trialName + ".trc")

#Setup OpenSim paths for later
openSimFolderName = 'OpenSimData'

openSimDir = os.path.join(sessionDir, openSimFolderName)        
KinematicsDataPath = os.path.join(openSimDir, 'Kinematics', trialName + ".mot")

# %% Trim down the TRC file

# Load in the TRC file
dataToTrim = osim.TimeSeriesTableVec3(markerDataPath)
#Delete any columns that end in "_study"
col_labels = dataToTrim.getColumnLabels()
    
#remove columns after the target index
for label in col_labels:
    if "_study" in label: 
        dataToTrim.removeColumn(label)

dataToTrim.trim(initialTime,finalTime)
osim.TRCFileAdapter().write(dataToTrim, pathOutputFiles[trialName])

# %% Calculate any vertical angle in the data and rotate

if rotateData:  
    kinematics_data = pd.read_table(KinematicsDataPath, delimiter='\t', skiprows=10, header=None)
    # Assuming the first row contains the column labels, set them as column names
    kinematics_data.columns = kinematics_data.iloc[0]
    # Drop the first row since it's now redundant (used as column names)
    kinematics_data = kinematics_data[1:]
    kinematics_data['time'] = pd.to_numeric(kinematics_data['time'], errors='coerce')  # Ensure time is numeric

    time_mask = (kinematics_data['time'] >= initialTime) & (kinematics_data['time'] <= finalTime)
    pelvis_ty_ROI = kinematics_data.loc[time_mask, 'pelvis_ty']  # Filter based on time window
    pelvis_tx_ROI = kinematics_data.loc[time_mask, 'pelvis_tx']
    
    # Extract the time and pelvis_ty values for the region of interest (ROI)
    time_ROI = kinematics_data.loc[time_mask, 'time'].astype(float)  # Convert to float
    pelvis_ty_ROI = kinematics_data.loc[time_mask, 'pelvis_ty'].astype(float)

    # Perform a linear fit (1st-degree polynomial fit)
    slope, intercept = np.polyfit(time_ROI, pelvis_ty_ROI, 1)    
    ty_start = slope*time_ROI.iloc[0] + intercept
    ty_end = slope*time_ROI.iloc[-1] + intercept

    # Print the slope and intercept of the linear fit
    #print(f"Slope of pelvis_ty with respect to time: {slope}")
        
    # # Get the start and end points for pelvis_ty and pelvis_tx
    # #get best fit not start/end of ty
    
    tx_start = float(pelvis_tx_ROI.iloc[0])
    tx_end = float(pelvis_tx_ROI.iloc[-1])
    
    # Calculate the slope using the start and end points
    slope_ROI = (ty_end - ty_start) / (tx_end - tx_start)
    # Calculate the arctangent in radians
    angle_radians = np.arctan(slope_ROI)
    # Convert radians to degrees
    angle = -np.degrees(angle_radians) #walking uphill needs a negative angle correction; downhill positive angle correction
    print(f"angle of correction (degrees): {angle}")
    
    # Use slope with Dataman to rotate the data
    
    dataTrimmedAndRotated = TRCFile(pathOutputFiles[trialName])

    dataTrimmedAndRotated.rotate('z', angle)

    # Save trimmed marker data to 
    dataTrimmedAndRotated.write(pathOutputFiles[trialName])

# %% Augmentation.
# Get augmenter model.
augmenterModelName = (
    sessionMetadata['markerAugmentationSettings']['markerAugmenterModel'])


postAugmentationDir = os.path.join(sessionDir, markerDataFolderName, 
                                   'PostAugmentation')

# Set output file name.
pathAugmentedOutputFiles = {}
pathAugmentedOutputFiles[trialName] = os.path.join(
        postAugmentationDir, trialName + ".trc")

#Rerun augmentation
os.makedirs(postAugmentationDir, exist_ok=True)    
augmenterDir = os.path.join(baseDir, "MarkerAugmenter")
logging.info('Augmenting marker set')
try:
    vertical_offset = augmentTRC(
        pathOutputFiles[trialName],sessionMetadata['mass_kg'], 
        sessionMetadata['height_m'], pathAugmentedOutputFiles[trialName],
        augmenterDir, augmenterModelName=augmenterModelName,
        augmenter_model=augmenterModel, offset=offset)
except Exception as e:
    if len(e.args) == 2: # specific exception
        raise Exception(e.args[0], e.args[1])
    elif len(e.args) == 1: # generic exception
        exception = "Marker augmentation failed. Verify your setup and try again. Visit https://www.opencap.ai/best-pratices to learn more about data collection and https://www.opencap.ai/troubleshooting for potential causes for a failed trial."
        raise Exception(exception, traceback.format_exc())

#TODO: Do we still need this?
if offset:
    # If offset, no need to offset again for the webapp visualization.
    # (0.01 so that there is no overall offset, see utilsOpenSim).
    vertical_offset_settings = float(np.copy(vertical_offset)-0.01)
    vertical_offset = 0.01   
    
# %% OpenSim pipeline.
if runOpenSimPipeline:
    openSimPipelineDir = os.path.join(baseDir, "opensimPipeline")        
     
    outputScaledModelDir = os.path.join(openSimDir, 'Model')

    # Check if shoulder model.
    if 'shoulder' in sessionMetadata['openSimModel']:
        suffix_model = '_shoulder'
    else:
        suffix_model = ''
    
    # Inverse kinematics.        
    outputIKDir = os.path.join(openSimDir, 'Kinematics/Trimmed')
    os.makedirs(outputIKDir, exist_ok=True)
    # Check if there is a scaled model.
    pathScaledModel = os.path.join(outputScaledModelDir, 
                                    sessionMetadata['openSimModel'] + 
                                    "_scaled.osim")
    if os.path.exists(pathScaledModel):
        # Path setup file.
        genericSetupFile4IKName = 'Setup_IK{}.xml'.format(suffix_model)
        pathGenericSetupFile4IK = os.path.join(
            openSimPipelineDir, 'IK', genericSetupFile4IKName)
        # Path TRC file.
        pathTRCFile4IK = pathAugmentedOutputFiles[trialName]
        # Run IK tool. 
        logging.info('Running Inverse Kinematics')
        try:
            pathOutputIK, pathModelIK = runIKTool(
                pathGenericSetupFile4IK, pathScaledModel, 
                pathTRCFile4IK, outputIKDir)
        except Exception as e:
            if len(e.args) == 2: # specific exception
                raise Exception(e.args[0], e.args[1])
            elif len(e.args) == 1: # generic exception
                exception = "Inverse kinematics failed. Verify your setup and try again. Visit https://www.opencap.ai/best-pratices to learn more about data collection and https://www.opencap.ai/troubleshooting for potential causes for a failed trial."
                raise Exception(exception, traceback.format_exc())
    else:
        raise ValueError("No scaled model available.")
            
    # Was supposed to get this from line 180 but it only had one output
    pathModelIK = pathScaledModel
    

    
##TODO: Is it necessary to rewrite these

# # Write body transforms to json for visualization.
# outputJsonVisDir = os.path.join(sessionDir,'VisualizerJsons',
#                                 trialName)
# os.makedirs(outputJsonVisDir,exist_ok=True)
# outputJsonVisPath = os.path.join(outputJsonVisDir,
#                                  trialName + '.json')
# generateVisualizerJson(pathModelIK, pathOutputIK,
#                        outputJsonVisPath, 
#                        vertical_offset=vertical_offset)  
    
# # Rewrite settings, adding offset  
#     if offset:
#         settings['verticalOffset'] = vertical_offset_settings 
#     with open(pathSettings, 'w') as file:
#         yaml.dump(settings, file)






