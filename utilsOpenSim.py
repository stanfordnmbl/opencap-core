import os
import utilsDataman
import opensim
import numpy as np
import glob
import json
from utils import storage2numpy

# %% Scaling.
def runScaleTool(pathGenericSetupFile, pathGenericModel, subjectMass,
                 pathTRCFile, timeRange, pathOutputFolder, 
                 scaledModelName='not_specified', subjectHeight=0,
                 createModelWithContacts=False, fixed_markers=False,
                 suffix_model=''):
    
    dirGenericModel, scaledModelNameA = os.path.split(pathGenericModel)
    
    # Paths.
    if scaledModelName == 'not_specified':
        scaledModelName = scaledModelNameA[:-5] + "_scaled"
    pathOutputModel = os.path.join(
        pathOutputFolder, scaledModelName + '.osim')
    pathOutputMotion = os.path.join(
        pathOutputFolder, scaledModelName + '.mot')
    pathOutputSetup =  os.path.join(
        pathOutputFolder, 'Setup_Scale_' + scaledModelName + '.xml')
    pathUpdGenericModel = os.path.join(
        pathOutputFolder, scaledModelNameA[:-5] + "_generic.osim")
    
    # Marker set.
    _, setupFileName = os.path.split(pathGenericSetupFile)
    if 'Lai' in scaledModelName or 'Rajagopal' in scaledModelName:
        if 'Mocap' in setupFileName:
            markerSetFileName = 'LaiUhlrich2022_markers_mocap{}.xml'.format(suffix_model)
        elif 'openpose' in setupFileName:
            markerSetFileName = 'LaiUhlrich2022_markers_openpose.xml'
        elif 'mmpose' in setupFileName:
            markerSetFileName = 'LaiUhlrich2022_markers_mmpose.xml'
        else:
            if fixed_markers:
                markerSetFileName = 'LaiUhlrich2022_markers_augmenter_fixed.xml'
            else:
                markerSetFileName = 'LaiUhlrich2022_markers_augmenter{}.xml'.format(suffix_model)
    elif 'gait2392' in scaledModelName:
         if 'Mocap' in setupFileName:
             markerSetFileName = 'gait2392_markers_mocap.xml'
         else:
            markerSetFileName = 'gait2392_markers_augmenter.xml'
    else:
        raise ValueError("Unknown model type: scaling")
    pathMarkerSet = os.path.join(dirGenericModel, markerSetFileName)
    
    # Add the marker set to the generic model and save that updated model.
    opensim.Logger.setLevelString('error')
    genericModel = opensim.Model(pathGenericModel)
    markerSet = opensim.MarkerSet(pathMarkerSet)
    genericModel.set_MarkerSet(markerSet)
    genericModel.printToXML(pathUpdGenericModel)    

    # Time range.
    timeRange_os = opensim.ArrayDouble(timeRange[0], 0)
    timeRange_os.insert(1, timeRange[-1])
                
    # Setup scale tool.
    scaleTool = opensim.ScaleTool(pathGenericSetupFile)
    scaleTool.setName(scaledModelName)
    scaleTool.setSubjectMass(subjectMass)
    scaleTool.setSubjectHeight(subjectHeight)
    genericModelMaker = scaleTool.getGenericModelMaker()     
    genericModelMaker.setModelFileName(pathUpdGenericModel)
    modelScaler = scaleTool.getModelScaler() 
    modelScaler.setMarkerFileName(pathTRCFile)
    modelScaler.setOutputModelFileName("")       
    modelScaler.setOutputScaleFileName("")
    modelScaler.setTimeRange(timeRange_os) 
    markerPlacer = scaleTool.getMarkerPlacer() 
    markerPlacer.setMarkerFileName(pathTRCFile)                
    markerPlacer.setOutputModelFileName(pathOutputModel)
    markerPlacer.setOutputMotionFileName(pathOutputMotion) 
    markerPlacer.setOutputMarkerFileName("")
    markerPlacer.setTimeRange(timeRange_os)
    
    # Disable tasks of dofs that are locked and markers that are not present.
    model = opensim.Model(pathUpdGenericModel)
    coordNames = []
    for coord in model.getCoordinateSet():
        if not coord.getDefaultLocked():
            coordNames.append(coord.getName())            
    modelMarkerNames = [marker.getName() for marker in model.getMarkerSet()]          
              
    for task in markerPlacer.getIKTaskSet():
        # Remove IK tasks for dofs that are locked or don't exist.
        if (task.getName() not in coordNames and 
            task.getConcreteClassName() == 'IKCoordinateTask'):
            task.setApply(False)
            print('{} is a locked coordinate - ignoring IK task'.format(
                task.getName()))
        # Remove Marker tracking tasks for markers not in model.
        if (task.getName() not in modelMarkerNames and 
            task.getConcreteClassName() == 'IKMarkerTask'):
            task.setApply(False)
            print('{} is not in model - ignoring IK task'.format(
                task.getName()))
            
    # Remove measurements from measurement set when markers don't exist.
    # Disable entire measurement if no complete marker pairs exist.
    measurementSet = modelScaler.getMeasurementSet()
    for meas in measurementSet:
        mkrPairSet = meas.getMarkerPairSet()
        iMkrPair = 0
        while iMkrPair < meas.getNumMarkerPairs():
            mkrPairNames = [
                mkrPairSet.get(iMkrPair).getMarkerName(i) for i in range(2)]
            if any([mkr not in modelMarkerNames for mkr in mkrPairNames]):
                mkrPairSet.remove(iMkrPair)
                print('{} or {} not in model. Removing associated \
                      MarkerPairSet from {}.'.format(mkrPairNames[0], 
                      mkrPairNames[1], meas.getName()))
            else:
                iMkrPair += 1
            if meas.getNumMarkerPairs() == 0:
                meas.setApply(False)
                print('There were no marker pairs in {}, so this measurement \
                      is not applied.'.format(meas.getName()))
    # Run scale tool.                      
    scaleTool.printToXML(pathOutputSetup)            
    command = 'opensim-cmd -o error' + ' run-tool ' + pathOutputSetup
    os.system(command)
    
    # Sanity check
    print(pathOutputModel)
    scaled_model = opensim.Model(pathOutputModel)
    bodySet = scaled_model.getBodySet()
    nBodies = bodySet.getSize()
    scale_factors = np.zeros((nBodies, 3))
    for i in range(nBodies):
        bodyName = bodySet.get(i).getName()
        body = bodySet.get(bodyName)
        attached_geometry = body.get_attached_geometry(0)
        scale_factors[i, :] = attached_geometry.get_scale_factors().to_numpy()
    diff_scale = np.max(np.max(scale_factors, axis=0)-
                        np.min(scale_factors, axis=0))
    # A difference in scaling factor larger than 1 would indicate that a 
    # segment (e.g., humerus) would be more than twice as large as its generic
    # counterpart, whereas another segment (e.g., pelvis) would have the same
    # size as the generic segment. This is very unlikely, but might occur when
    # the camera calibration went wrong (i.e., bad extrinsics).
    if diff_scale > 1:
        exception = "Musculoskeletal model scaling failed; the segment sizes are not anthropometrically realistic. It is very likely that the camera calibration went wrong. Visit https://www.opencap.ai/best-pratices to learn more about camera calibration."
        raise Exception(exception, exception)        
    
    return pathOutputModel
    
# %% Inverse kinematics.
def runIKTool(pathGenericSetupFile, pathScaledModel, pathTRCFile,
              pathOutputFolder, timeRange=[], IKFileName='not_specified'):
    
    # Paths
    if IKFileName == 'not_specified':
        _, IKFileName = os.path.split(pathTRCFile)
        IKFileName = IKFileName[:-4]
    pathOutputMotion = os.path.join(
        pathOutputFolder, IKFileName + '.mot')
    pathOutputSetup =  os.path.join(
        pathOutputFolder, 'Setup_IK_' + IKFileName + '.xml')
    
    # To make IK faster, we remove the patellas and their constraints from the
    # model. Constraints make the IK problem more difficult, and the patellas
    # are not used in the IK solution for this particular model. Since muscles
    # are attached to the patellas, we also remove all muscles.
    opensim.Logger.setLevelString('error')
    model = opensim.Model(pathScaledModel)
    # Remove all actuators.                                         
    forceSet = model.getForceSet()
    forceSet.setSize(0)
    # Remove patellofemoral constraints.
    constraintSet = model.getConstraintSet()
    patellofemoral_constraints = [
        'patellofemoral_knee_angle_r_con', 'patellofemoral_knee_angle_l_con']
    for patellofemoral_constraint in patellofemoral_constraints:
        i = constraintSet.getIndex(patellofemoral_constraint, 0)
        constraintSet.remove(i)       
    # Remove patella bodies.
    bodySet = model.getBodySet()
    patella_bodies = ['patella_r', 'patella_l']
    for patella in patella_bodies:
        i = bodySet.getIndex(patella, 0)
        bodySet.remove(i)
    # Remove patellofemoral joints.
    jointSet = model.getJointSet()
    patellofemoral_joints = ['patellofemoral_r', 'patellofemoral_l']
    for patellofemoral in patellofemoral_joints:
        i = jointSet.getIndex(patellofemoral, 0)
        jointSet.remove(i)
    # Print the model to a new file.
    model.finalizeConnections
    model.initSystem()
    pathScaledModelWithoutPatella = pathScaledModel.replace('.osim', '_no_patella.osim')
    model.printToXML(pathScaledModelWithoutPatella)   

    # Setup IK tool.    
    IKTool = opensim.InverseKinematicsTool(pathGenericSetupFile)            
    IKTool.setName(IKFileName)
    IKTool.set_model_file(pathScaledModelWithoutPatella)          
    IKTool.set_marker_file(pathTRCFile)
    if timeRange:
        IKTool.set_time_range(0, timeRange[0])
        IKTool.set_time_range(1, timeRange[-1])
    IKTool.setResultsDir(pathOutputFolder)                        
    IKTool.set_report_errors(True)
    IKTool.set_report_marker_locations(False)
    IKTool.set_output_motion_file(pathOutputMotion)
    IKTool.printToXML(pathOutputSetup)
    command = 'opensim-cmd -o error' + ' run-tool ' + pathOutputSetup
    os.system(command)
    
    return pathOutputMotion, pathScaledModelWithoutPatella
    
    
# %% This function will look for a time window, of a minimum duration specified
# by thresholdTime, during which the markers move at most by a distance
# specified by thresholdPosition.
def getScaleTimeRange(pathTRCFile, thresholdPosition=0.005, thresholdTime=0.3,
                      withArms=True, withOpenPoseMarkers=False, isMocap=False,
                      removeRoot=False):
    
    c_trc_file = utilsDataman.TRCFile(pathTRCFile)
    c_trc_time = c_trc_file.time    
    if withOpenPoseMarkers:
        # No big toe markers, such as to include both OpenPose and mmpose.
        markers = ["Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RKnee", 
                   "LKnee", "RAnkle", "LAnkle", "RHeel", "LHeel", "RSmallToe", 
                   "LSmallToe", "RElbow", "LElbow", "RWrist", "LWrist"]        
    else:
        markers = ["C7_study", "r_shoulder_study", "L_shoulder_study",
                   "r.ASIS_study", "L.ASIS_study", "r.PSIS_study", 
                   "L.PSIS_study", "r_knee_study", "L_knee_study",
                   "r_mknee_study", "L_mknee_study", "r_ankle_study", 
                   "L_ankle_study", "r_mankle_study", "L_mankle_study",
                   "r_calc_study", "L_calc_study", "r_toe_study", 
                   "L_toe_study", "r_5meta_study", "L_5meta_study",
                   "RHJC_study", "LHJC_study"]
        if withArms:
            markers.append("r_lelbow_study")
            markers.append("L_lelbow_study")
            markers.append("r_melbow_study")
            markers.append("L_melbow_study")
            markers.append("r_lwrist_study")
            markers.append("L_lwrist_study")
            markers.append("r_mwrist_study")
            markers.append("L_mwrist_study")
            
        if isMocap:
            markers = [marker.replace('_study','') for marker in markers]
            markers = [marker.replace('r_shoulder','R_Shoulder') for marker in markers] # should just change the mocap marker set
            markers = [marker.replace('L_shoulder','L_Shoulder') for marker in markers] # should just change the mocap marker set
            markers = [marker.replace('RHJC','R_HJC') for marker in markers] # should just change the mocap marker set
            markers = [marker.replace('LHJC','L_HJC') for marker in markers] # should just change the mocap marker set
            markers = [marker.replace('r_lelbow','R_elbow_lat') for marker in markers] # should just change the mocap marker set
            markers = [marker.replace('L_lelbow','L_elbow_lat') for marker in markers] # should just change the mocap marker set
            markers = [marker.replace('r_melbow','R_elbow_med') for marker in markers] # should just change the mocap marker set
            markers = [marker.replace('L_melbow','L_elbow_med') for marker in markers] # should just change the mocap marker set
            markers = [marker.replace('r_lwrist','R_wrist_radius') for marker in markers] # should just change the mocap marker set
            markers = [marker.replace('L_lwrist','L_wrist_radius') for marker in markers] # should just change the mocap marker set
            markers = [marker.replace('r_mwrist','R_wrist_ulna') for marker in markers] # should just change the mocap marker set
            markers = [marker.replace('L_mwrist','L_wrist_ulna') for marker in markers] # should just change the mocap marker set
            

    trc_data = np.zeros((c_trc_time.shape[0], 3*len(markers)))
    for count, marker in enumerate(markers):
        trc_data[:, count*3:count*3+3] = c_trc_file.marker(marker)
    
    if removeRoot:
        try:
            root_data = c_trc_file.marker('midHip')
            trc_data -= np.tile(root_data,len(markers))
        except:
            pass
   
    if np.max(trc_data)>10: # in mm, turn to m
        trc_data/=1000
        
    # Sampling frequency.
    sf = np.round(1/np.mean(np.diff(c_trc_time)),4)
    # Minimum duration for time range in seconds.
    timeRange_min = 1
    # Corresponding number of frames.
    nf = int(timeRange_min*sf + 1)
    
    detectedWindow = False
    i = 0
    while not detectedWindow:
        c_window = trc_data[i:i+nf,:]
        c_window_max = np.max(c_window, axis=0)
        c_window_min = np.min(c_window, axis=0)
        c_window_diff = np.abs(c_window_max - c_window_min)
        detectedWindow = np.alltrue(c_window_diff<thresholdPosition)
        if not detectedWindow:
            i += 1
            if i > c_trc_time.shape[0]-nf:
                i = 0
                nf -= int(0.1*sf) 
            if np.round((nf-1)/sf,2) < thresholdTime: # number of frames got too small without detecting a window
                exception = "Musculoskeletal model scaling failed; could not detect a static phase of at least %.2fs. After you press record, make sure the subject stands still until the message tells you they can relax . Visit https://www.opencap.ai/best-pratices to learn more about data collection." % thresholdTime
                raise Exception(exception, exception)
    
    timeRange = [c_trc_time[i], c_trc_time[i+nf-1]]
    timeRangeSpan = np.round(timeRange[1] - timeRange[0], 2)
    
    print("Static phase of %.2fs detected in staticPose between [%.2f, %.2f]."
          % (timeRangeSpan,np.round(timeRange[0],2),np.round(timeRange[1],2)))
 
    return timeRange

# %%
def runIDTool(pathGenericSetupFileID, pathGenericSetupFileEL, pathGRFFile,              
              pathScaledModel, pathIKFile, timeRange, pathOutputFolder,
              filteringFrequency=10, IKFileName='not_specified'):
    
    # Paths
    if IKFileName == 'not_specified':
        _, IKFileName = os.path.split(pathIKFile)
        IKFileName = IKFileName[:-4]
        
    pathOutputSetupEL =  os.path.join(
        pathOutputFolder, 'Setup_EL_' + IKFileName + '.xml')
    pathOutputSetupID =  os.path.join(
        pathOutputFolder, 'Setup_ID_' + IKFileName + '.xml')
    
    # External loads
    opensim.Logger.setLevelString('error')
    ELTool = opensim.ExternalLoads(pathGenericSetupFileEL, True)
    ELTool.setDataFileName(pathGRFFile)
    ELTool.setName(IKFileName)
    ELTool.printToXML(pathOutputSetupEL)
    
    # ID    
    IDTool = opensim.InverseDynamicsTool(pathGenericSetupFileID)
    IDTool.setModelFileName(pathScaledModel)
    IDTool.setName(IKFileName)
    IDTool.setStartTime(timeRange[0])
    IDTool.setEndTime(timeRange[-1])      
    IDTool.setExternalLoadsFileName(pathOutputSetupEL)
    IDTool.setCoordinatesFileName(pathIKFile)
    IDTool.setLowpassCutoffFrequency(filteringFrequency)
    IDTool.setResultsDir(pathOutputFolder)
    IDTool.setOutputGenForceFileName(IKFileName + '.sto')   
    IDTool.printToXML(pathOutputSetupID)   
    command = 'opensim-cmd -o error' + ' run-tool ' + pathOutputSetupID
    os.system(command)

# %% Might be outdated.
def addOpenPoseMarkersTool(pathModel, adjustLocationHipAnkle=True, 
                           hipOffsetX=0.036, hipOffsetZ=0.0175, 
                           ankleOffsetX=0.007):
    
    '''
        This script adds virtual markers corresponding to the OpenPose markers
        to the OpenSim model. For most markers, we assume that the OpenPose
        markers correspond to the joint centers. For some markers, the
        locations have been hand-tuned for one subject.
    '''
    
    pathModelFolder, modelName = os.path.split(pathModel)
    
    # Methods specific to each marker.
    markersInfo = {
            "RHip": {"method": "jointCenter", "joint": "hip_r", "parent": "pelvis"},
            "LHip": {"method": "jointCenter", "joint": "hip_l", "parent": "pelvis"},
            "midHip": {"method": "mid", "references": "jointCenters", "reference_jointCenter_A": "hip_r", "reference_jointCenter_B": "hip_l"},
            "RElbow": {"method": "jointCenter", "joint": "elbow_r", "parent": "humerus_r"},
            "LElbow": {"method": "jointCenter", "joint": "elbow_l", "parent": "humerus_l"},
            "RWrist": {"method": "jointCenter", "joint": "radius_hand_r", "parent": "hand_r"},
            "LWrist": {"method": "jointCenter", "joint": "radius_hand_l", "parent": "hand_l"},
            "RShoulder": {"method": "jointCenter", "joint": "acromial_r", "parent": "torso"},
            "LShoulder": {"method": "jointCenter", "joint": "acromial_l", "parent": "torso"},
            "RKnee": {"method": "jointCenter", "joint": "walker_knee_r", "parent": "tibia_r"},
            "LKnee": {"method": "jointCenter", "joint": "walker_knee_l", "parent": "tibia_l"},
            "Neck": {"method": "mid", "references": "jointCenters", "reference_jointCenter_A": "acromial_r", "reference_jointCenter_B": "acromial_l"}}
    # Values manually set.
    markersInfo["RHeel"] = {"method": "location", "parent": "calcn_r", "location": np.array([0.018205985755086355, 0.01, -0.020246741086146904])}
    markersInfo["LHeel"] = {"method": "location", "parent": "calcn_l", "location": np.array([0.018205985755086355, 0.01, 0.020246741086146904])}
    markersInfo["RSmallToe"] = {"method": "location", "parent": "toes_r", "location": np.array([0.0214658, 0.002, 0.0394135])}
    markersInfo["LSmallToe"] = {"method": "location", "parent": "toes_l", "location": np.array([0.0214658, 0.002, -0.0394135])}
    markersInfo["RBigToe"] = {"method": "location", "parent": "toes_r", "location": np.array([0.0487748, 0.002, -0.0162651])}
    markersInfo["LBigToe"] = {"method": "location", "parent": "toes_l", "location": np.array([0.0487748, 0.002, 0.0162651])} 
    markersInfo["RAnkle"] = {"method": "location", "parent": "tibia_r", "location": np.array([-0.007, -0.38, 0.0075])}
    markersInfo["LAnkle"] = {"method": "location", "parent": "tibia_l", "location": np.array([-0.007, -0.38, -0.0075])}
    
    # Add OpenPose markers and print new model
    opensim.Logger.setLevelString('error')
    model = opensim.Model(pathModel)    
    bodySet = model.get_BodySet() 
    jointSet = model.get_JointSet()  
    markerSet = model.get_MarkerSet()  
    for marker in markersInfo:
        
        if markersInfo[marker]["method"] == "marker":
            referenceMarker = markerSet.get(markersInfo[marker]["reference_marker"])            
            parentFrame = referenceMarker.getParentFrameName()
            location = referenceMarker.get_location()
            
        if markersInfo[marker]["method"] == "jointCenter":
                joint = jointSet.get(markersInfo[marker]["joint"])            
                if ((marker == "RKnee" and markersInfo[marker]["parent"] == "tibia_r") or  
                    (marker == "LKnee" and markersInfo[marker]["parent"] == "tibia_l") or
                    (marker == "RKnee" and markersInfo[marker]["parent"] == "sagittal_articulation_frame_r") or
                    (marker == "LKnee" and markersInfo[marker]["parent"] == "sagittal_articulation_frame_l") or
                    (marker == "RWrist" and markersInfo[marker]["parent"] == "hand_r") or
                    (marker == "LWrist" and markersInfo[marker]["parent"] == "hand_l")):
                    frame = joint.get_frames(1)
                else:
                    frame = joint.get_frames(0)
                assert frame.getName()[:-7] == markersInfo[marker]["parent"]
                location = frame.get_translation()
                
                if adjustLocationHipAnkle:
                    if marker == "RHip" or marker == "LHip":
                        
                        # Get scale factor based on saved factors from scaling.
                        body = bodySet.get(markersInfo[marker]["parent"])
                        attached_geometry = body.get_attached_geometry(0)
                        scale_factors = attached_geometry.get_scale_factors().to_numpy()
                        location_np = location.to_numpy()
                        
                        # After comparing triangulated OpenPose markers and
                        # mocap-based markers, it appears that the hip OpenPose
                        # markers should be located more forward and lateral.
                        location_adj_np = np.copy(location_np)
                        location_adj_np[0] += (hipOffsetX * scale_factors[0])
                        if marker == "RHip":
                            location_adj_np[2] += (hipOffsetZ * scale_factors[2])
                        elif marker == "LHip":
                            location_adj_np[2] -= (hipOffsetZ * scale_factors[2])
                        
                        location = opensim.Vec3(location_adj_np)
                
                parentFrame = "/bodyset/" + markersInfo[marker][import os
import utilsDataman
import opensim
import numpy as np
import glob
import json
from utils import storage2numpy

# %% Scaling.
def runScaleTool(pathGenericSetupFile, pathGenericModel, subjectMass,
                 pathTRCFile, timeRange, pathOutputFolder, 
                 scaledModelName='not_specified', subjectHeight=0,
                 createModelWithContacts=False, fixed_markers=False,
                 suffix_model=''):
    
    dirGenericModel, scaledModelNameA = os.path.split(pathGenericModel)
    
    # Paths.
    if scaledModelName == 'not_specified':
        scaledModelName = scaledModelNameA[:-5] + "_scaled"
    pathOutputModel = os.path.join(
        pathOutputFolder, scaledModelName + '.osim')
    pathOutputMotion = os.path.join(
        pathOutputFolder, scaledModelName + '.mot')
    pathOutputSetup =  os.path.join(
        pathOutputFolder, 'Setup_Scale_' + scaledModelName + '.xml')
    pathUpdGenericModel = os.path.join(
        pathOutputFolder, scaledModelNameA[:-5] + "_generic.osim")
    
    # Marker set.
    _, setupFileName = os.path.split(pathGenericSetupFile)
    if 'Lai' in scaledModelName or 'Rajagopal' in scaledModelName:
        if 'Mocap' in setupFileName:
            markerSetFileName = 'LaiUhlrich2022_markers_mocap{}.xml'.format(suffix_model)
        elif 'openpose' in setupFileName:
            markerSetFileName = 'LaiUhlrich2022_markers_openpose.xml'
        elif 'mmpose' in setupFileName:
            markerSetFileName = 'LaiUhlrich2022_markers_mmpose.xml'
        else:
            if fixed_markers:
                markerSetFileName = 'LaiUhlrich2022_markers_augmenter_fixed.xml'
            else:
                markerSetFileName = 'LaiUhlrich2022_markers_augmenter{}.xml'.format(suffix_model)
    elif 'gait2392' in scaledModelName:
         if 'Mocap' in setupFileName:
             markerSetFileName = 'gait2392_markers_mocap.xml'
         else:
            markerSetFileName = 'gait2392_markers_augmenter.xml'
    else:
        raise ValueError("Unknown model type: scaling")
    pathMarkerSet = os.path.join(dirGenericModel, markerSetFileName)
    
    # Add the marker set to the generic model and save that updated model.
    opensim.Logger.setLevelString('error')
    genericModel = opensim.Model(pathGenericModel)
    markerSet = opensim.MarkerSet(pathMarkerSet)
    genericModel.set_MarkerSet(markerSet)
    genericModel.printToXML(pathUpdGenericModel)    

    # Time range.
    timeRange_os = opensim.ArrayDouble(timeRange[0], 0)
    timeRange_os.insert(1, timeRange[-1])
                
    # Setup scale tool.
    scaleTool = opensim.ScaleTool(pathGenericSetupFile)
    scaleTool.setName(scaledModelName)
    scaleTool.setSubjectMass(subjectMass)
    scaleTool.setSubjectHeight(subjectHeight)
    genericModelMaker = scaleTool.getGenericModelMaker()     
    genericModelMaker.setModelFileName(pathUpdGenericModel)
    modelScaler = scaleTool.getModelScaler() 
    modelScaler.setMarkerFileName(pathTRCFile)
    modelScaler.setOutputModelFileName("")       
    modelScaler.setOutputScaleFileName("")
    modelScaler.setTimeRange(timeRange_os) 
    markerPlacer = scaleTool.getMarkerPlacer() 
    markerPlacer.setMarkerFileName(pathTRCFile)                
    markerPlacer.setOutputModelFileName(pathOutputModel)
    markerPlacer.setOutputMotionFileName(pathOutputMotion) 
    markerPlacer.setOutputMarkerFileName("")
    markerPlacer.setTimeRange(timeRange_os)
    
    # Disable tasks of dofs that are locked and markers that are not present.
    model = opensim.Model(pathUpdGenericModel)
    coordNames = []
    for coord in model.getCoordinateSet():
        if not coord.getDefaultLocked():
            coordNames.append(coord.getName())            
    modelMarkerNames = [marker.getName() for marker in model.getMarkerSet()]          
              
    for task in markerPlacer.getIKTaskSet():
        # Remove IK tasks for dofs that are locked or don't exist.
        if (task.getName() not in coordNames and 
            task.getConcreteClassName() == 'IKCoordinateTask'):
            task.setApply(False)
            print('{} is a locked coordinate - ignoring IK task'.format(
                task.getName()))
        # Remove Marker tracking tasks for markers not in model.
        if (task.getName() not in modelMarkerNames and 
            task.getConcreteClassName() == 'IKMarkerTask'):
            task.setApply(False)
            print('{} is not in model - ignoring IK task'.format(
                task.getName()))
            
    # Remove measurements from measurement set when markers don't exist.
    # Disable entire measurement if no complete marker pairs exist.
    measurementSet = modelScaler.getMeasurementSet()
    for meas in measurementSet:
        mkrPairSet = meas.getMarkerPairSet()
        iMkrPair = 0
        while iMkrPair < meas.getNumMarkerPairs():
            mkrPairNames = [
                mkrPairSet.get(iMkrPair).getMarkerName(i) for i in range(2)]
            if any([mkr not in modelMarkerNames for mkr in mkrPairNames]):
                mkrPairSet.remove(iMkrPair)
                print('{} or {} not in model. Removing associated \
                      MarkerPairSet from {}.'.format(mkrPairNames[0], 
                      mkrPairNames[1], meas.getName()))
            else:
                iMkrPair += 1
            if meas.getNumMarkerPairs() == 0:
                meas.setApply(False)
                print('There were no marker pairs in {}, so this measurement \
                      is not applied.'.format(meas.getName()))
    # Run scale tool.                      
    scaleTool.printToXML(pathOutputSetup)            
    command = 'opensim-cmd -o error' + ' run-tool ' + pathOutputSetup
    os.system(command)
    
    # Sanity check
    print(pathOutputModel)
    scaled_model = opensim.Model(pathOutputModel)
    bodySet = scaled_model.getBodySet()
    nBodies = bodySet.getSize()
    scale_factors = np.zeros((nBodies, 3))
    for i in range(nBodies):
        bodyName = bodySet.get(i).getName()
        body = bodySet.get(bodyName)
        attached_geometry = body.get_attached_geometry(0)
        scale_factors[i, :] = attached_geometry.get_scale_factors().to_numpy()
    diff_scale = np.max(np.max(scale_factors, axis=0)-
                        np.min(scale_factors, axis=0))
    # A difference in scaling factor larger than 1 would indicate that a 
    # segment (e.g., humerus) would be more than twice as large as its generic
    # counterpart, whereas another segment (e.g., pelvis) would have the same
    # size as the generic segment. This is very unlikely, but might occur when
    # the camera calibration went wrong (i.e., bad extrinsics).
    if diff_scale > 1:
        exception = "Musculoskeletal model scaling failed; the segment sizes are not anthropometrically realistic. It is very likely that the camera calibration went wrong. Visit https://www.opencap.ai/best-pratices to learn more about camera calibration."
        raise Exception(exception, exception)        
    
    return pathOutputModel
    
# %% Inverse kinematics.
def runIKTool(pathGenericSetupFile, pathScaledModel, pathTRCFile,
              pathOutputFolder, timeRange=[], IKFileName='not_specified'):
    
    # Paths
    if IKFileName == 'not_specified':
        _, IKFileName = os.path.split(pathTRCFile)
        IKFileName = IKFileName[:-4]
    pathOutputMotion = os.path.join(
        pathOutputFolder, IKFileName + '.mot')
    pathOutputSetup =  os.path.join(
        pathOutputFolder, 'Setup_IK_' + IKFileName + '.xml')
    
    # To make IK faster, we remove the patellas and their constraints from the
    # model. Constraints make the IK problem more difficult, and the patellas
    # are not used in the IK solution for this particular model. Since muscles
    # are attached to the patellas, we also remove all muscles.
    opensim.Logger.setLevelString('error')
    model = opensim.Model(pathScaledModel)
    # Remove all actuators.                                         
    forceSet = model.getForceSet()
    forceSet.setSize(0)
    # Remove patellofemoral constraints.
    constraintSet = model.getConstraintSet()
    patellofemoral_constraints = [
        'patellofemoral_knee_angle_r_con', 'patellofemoral_knee_angle_l_con']
    for patellofemoral_constraint in patellofemoral_constraints:
        i = constraintSet.getIndex(patellofemoral_constraint, 0)
        constraintSet.remove(i)       
    # Remove patella bodies.
    bodySet = model.getBodySet()
    patella_bodies = ['patella_r', 'patella_l']
    for patella in patella_bodies:
        i = bodySet.getIndex(patella, 0)
        bodySet.remove(i)
    # Remove patellofemoral joints.
    jointSet = model.getJointSet()
    patellofemoral_joints = ['patellofemoral_r', 'patellofemoral_l']
    for patellofemoral in patellofemoral_joints:
        i = jointSet.getIndex(patellofemoral, 0)
        jointSet.remove(i)
    # Print the model to a new file.
    model.finalizeConnections
    model.initSystem()
    pathScaledModelWithoutPatella = pathScaledModel.replace('.osim', '_no_patella.osim')
    model.printToXML(pathScaledModelWithoutPatella)   

    # Setup IK tool.    
    IKTool = opensim.InverseKinematicsTool(pathGenericSetupFile)            
    IKTool.setName(IKFileName)
    IKTool.set_model_file(pathScaledModelWithoutPatella)          
    IKTool.set_marker_file(pathTRCFile)
    if timeRange:
        IKTool.set_time_range(0, timeRange[0])
        IKTool.set_time_range(1, timeRange[-1])
    IKTool.setResultsDir(pathOutputFolder)                        
    IKTool.set_report_errors(True)
    IKTool.set_report_marker_locations(False)
    IKTool.set_output_motion_file(pathOutputMotion)
    IKTool.printToXML(pathOutputSetup)
    command = 'opensim-cmd -o error' + ' run-tool ' + pathOutputSetup
    os.system(command)
    
    return pathOutputMotion, pathScaledModelWithoutPatella
    
    
# %% This function will look for a time window, of a minimum duration specified
# by thresholdTime, during which the markers move at most by a distance
# specified by thresholdPosition.
def getScaleTimeRange(pathTRCFile, thresholdPosition=0.005, thresholdTime=0.3,
                      withArms=True, withOpenPoseMarkers=False, isMocap=False,
                      removeRoot=False):
    
    c_trc_file = utilsDataman.TRCFile(pathTRCFile)
    c_trc_time = c_trc_file.time    
    if withOpenPoseMarkers:
        # No big toe markers, such as to include both OpenPose and mmpose.
        markers = ["Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RKnee", 
                   "LKnee", "RAnkle", "LAnkle", "RHeel", "LHeel", "RSmallToe", 
                   "LSmallToe", "RElbow", "LElbow", "RWrist", "LWrist"]        
    else:
        markers = ["C7_study", "r_shoulder_study", "L_shoulder_study",
                   "r.ASIS_study", "L.ASIS_study", "r.PSIS_study", 
                   "L.PSIS_study", "r_knee_study", "L_knee_study",
                   "r_mknee_study", "L_mknee_study", "r_ankle_study", 
                   "L_ankle_study", "r_mankle_study", "L_mankle_study",
                   "r_calc_study", "L_calc_study", "r_toe_study", 
                   "L_toe_study", "r_5meta_study", "L_5meta_study",
                   "RHJC_study", "LHJC_study"]
        if withArms:
            markers.append("r_lelbow_study")
            markers.append("L_lelbow_study")
            markers.append("r_melbow_study")
            markers.append("L_melbow_study")
            markers.append("r_lwrist_study")
            markers.append("L_lwrist_study")
            markers.append("r_mwrist_study")
            markers.append("L_mwrist_study")
            
        if isMocap:
            markers = [marker.replace('_study','') for marker in markers]
            markers = [marker.replace('r_shoulder','R_Shoulder') for marker in markers] # should just change the mocap marker set
            markers = [marker.replace('L_shoulder','L_Shoulder') for marker in markers] # should just change the mocap marker set
            markers = [marker.replace('RHJC','R_HJC') for marker in markers] # should just change the mocap marker set
            markers = [marker.replace('LHJC','L_HJC') for marker in markers] # should just change the mocap marker set
            markers = [marker.replace('r_lelbow','R_elbow_lat') for marker in markers] # should just change the mocap marker set
            markers = [marker.replace('L_lelbow','L_elbow_lat') for marker in markers] # should just change the mocap marker set
            markers = [marker.replace('r_melbow','R_elbow_med') for marker in markers] # should just change the mocap marker set
            markers = [marker.replace('L_melbow','L_elbow_med') for marker in markers] # should just change the mocap marker set
            markers = [marker.replace('r_lwrist','R_wrist_radius') for marker in markers] # should just change the mocap marker set
            markers = [marker.replace('L_lwrist','L_wrist_radius') for marker in markers] # should just change the mocap marker set
            markers = [marker.replace('r_mwrist','R_wrist_ulna') for marker in markers] # should just change the mocap marker set
            markers = [marker.replace('L_mwrist','L_wrist_ulna') for marker in markers] # should just change the mocap marker set
            

    trc_data = np.zeros((c_trc_time.shape[0], 3*len(markers)))
    for count, marker in enumerate(markers):
        trc_data[:, count*3:count*3+3] = c_trc_file.marker(marker)
    
    if removeRoot:
        try:
            root_data = c_trc_file.marker('midHip')
            trc_data -= np.tile(root_data,len(markers))
        except:
            pass
   
    if np.max(trc_data)>10: # in mm, turn to m
        trc_data/=1000
        
    # Sampling frequency.
    sf = np.round(1/np.mean(np.diff(c_trc_time)),4)
    # Minimum duration for time range in seconds.
    timeRange_min = 1
    # Corresponding number of frames.
    nf = int(timeRange_min*sf + 1)
    
    detectedWindow = False
    i = 0
    while not detectedWindow:
        c_window = trc_data[i:i+nf,:]
        c_window_max = np.max(c_window, axis=0)
        c_window_min = np.min(c_window, axis=0)
        c_window_diff = np.abs(c_window_max - c_window_min)
        detectedWindow = np.alltrue(c_window_diff<thresholdPosition)
        if not detectedWindow:
            i += 1
            if i > c_trc_time.shape[0]-nf:
                i = 0
                nf -= int(0.1*sf) 
            if np.round((nf-1)/sf,2) < thresholdTime: # number of frames got too small without detecting a window
                exception = "Musculoskeletal model scaling failed; could not detect a static phase of at least %.2fs. After you press record, make sure the subject stands still until the message tells you they can relax . Visit https://www.opencap.ai/best-pratices to learn more about data collection." % thresholdTime
                raise Exception(exception, exception)
    
    timeRange = [c_trc_time[i], c_trc_time[i+nf-1]]
    timeRangeSpan = np.round(timeRange[1] - timeRange[0], 2)
    
    print("Static phase of %.2fs detected in staticPose between [%.2f, %.2f]."
          % (timeRangeSpan,np.round(timeRange[0],2),np.round(timeRange[1],2)))
 
    return timeRange

# %%
def runIDTool(pathGenericSetupFileID, pathGenericSetupFileEL, pathGRFFile,              
              pathScaledModel, pathIKFile, timeRange, pathOutputFolder,
              filteringFrequency=10, IKFileName='not_specified'):
    
    # Paths
    if IKFileName == 'not_specified':
        _, IKFileName = os.path.split(pathIKFile)
        IKFileName = IKFileName[:-4]
        
    pathOutputSetupEL =  os.path.join(
        pathOutputFolder, 'Setup_EL_' + IKFileName + '.xml')
    pathOutputSetupID =  os.path.join(
        pathOutputFolder, 'Setup_ID_' + IKFileName + '.xml')
    
    # External loads
    opensim.Logger.setLevelString('error')
    ELTool = opensim.ExternalLoads(pathGenericSetupFileEL, True)
    ELTool.setDataFileName(pathGRFFile)
    ELTool.setName(IKFileName)
    ELTool.printToXML(pathOutputSetupEL)
    
    # ID    
    IDTool = opensim.InverseDynamicsTool(pathGenericSetupFileID)
    IDTool.setModelFileName(pathScaledModel)
    IDTool.setName(IKFileName)
    IDTool.setStartTime(timeRange[0])
    IDTool.setEndTime(timeRange[-1])      
    IDTool.setExternalLoadsFileName(pathOutputSetupEL)
    IDTool.setCoordinatesFileName(pathIKFile)
    IDTool.setLowpassCutoffFrequency(filteringFrequency)
    IDTool.setResultsDir(pathOutputFolder)
    IDTool.setOutputGenForceFileName(IKFileName + '.sto')   
    IDTool.printToXML(pathOutputSetupID)   
    command = 'opensim-cmd -o error' + ' run-tool ' + pathOutputSetupID
    os.system(command)

# %% Might be outdated.
def addOpenPoseMarkersTool(pathModel, adjustLocationHipAnkle=True, 
                           hipOffsetX=0.036, hipOffsetZ=0.0175, 
                           ankleOffsetX=0.007):
    
    '''
        This script adds virtual markers corresponding to the OpenPose markers
        to the OpenSim model. For most markers, we assume that the OpenPose
        markers correspond to the joint centers. For some markers, the
        locations have been hand-tuned for one subject.
    '''
    
    pathModelFolder, modelName = os.path.split(pathModel)
    
    # Methods specific to each marker.
    markersInfo = {
            "RHip": {"method": "jointCenter", "joint": "hip_r", "parent": "pelvis"},
            "LHip": {"method": "jointCenter", "joint": "hip_l", "parent": "pelvis"},
            "midHip": {"method": "mid", "references": "jointCenters", "reference_jointCenter_A": "hip_r", "reference_jointCenter_B": "hip_l"},
            "RElbow": {"method": "jointCenter", "joint": "elbow_r", "parent": "humerus_r"},
            "LElbow": {"method": "jointCenter", "joint": "elbow_l", "parent": "humerus_l"},
            "RWrist": {"method": "jointCenter", "joint": "radius_hand_r", "parent": "hand_r"},
            "LWrist": {"method": "jointCenter", "joint": "radius_hand_l", "parent": "hand_l"},
            "RShoulder": {"method": "jointCenter", "joint": "acromial_r", "parent": "torso"},
            "LShoulder": {"method": "jointCenter", "joint": "acromial_l", "parent": "torso"},
            "RKnee": {"method": "jointCenter", "joint": "walker_knee_r", "parent": "tibia_r"},
            "LKnee": {"method": "jointCenter", "joint": "walker_knee_l", "parent": "tibia_l"},
            "Neck": {"method": "mid", "references": "jointCenters", "reference_jointCenter_A": "acromial_r", "reference_jointCenter_B": "acromial_l"}}
    # Values manually set.
    markersInfo["RHeel"] = {"method": "location", "parent": "calcn_r", "location": np.array([0.018205985755086355, 0.01, -0.020246741086146904])}
    markersInfo["LHeel"] = {"method": "location", "parent": "calcn_l", "location": np.array([0.018205985755086355, 0.01, 0.020246741086146904])}
    markersInfo["RSmallToe"] = {"method": "location", "parent": "toes_r", "location": np.array([0.0214658, 0.002, 0.0394135])}
    markersInfo["LSmallToe"] = {"method": "location", "parent": "toes_l", "location": np.array([0.0214658, 0.002, -0.0394135])}
    markersInfo["RBigToe"] = {"method": "location", "parent": "toes_r", "location": np.array([0.0487748, 0.002, -0.0162651])}
    markersInfo["LBigToe"] = {"method": "location", "parent": "toes_l", "location": np.array([0.0487748, 0.002, 0.0162651])} 
    markersInfo["RAnkle"] = {"method": "location", "parent": "tibia_r", "location": np.array([-0.007, -0.38, 0.0075])}
    markersInfo["LAnkle"] = {"method": "location", "parent": "tibia_l", "location": np.array([-0.007, -0.38, -0.0075])}
    
    # Add OpenPose markers and print new model
    opensim.Logger.setLevelString('error')
    model = opensim.Model(pathModel)    
    bodySet = model.get_BodySet() 
    jointSet = model.get_JointSet()  
    markerSet = model.get_MarkerSet()  
    for marker in markersInfo:
        
        if markersInfo[marker]["method"] == "marker":
            referenceMarker = markerSet.get(markersInfo[marker]["reference_marker"])            
            parentFrame = referenceMarker.getParentFrameName()
            location = referenceMarker.get_location()
            
        if markersInfo[marker]["method"] == "jointCenter":
                joint = jointSet.get(markersInfo[marker]["joint"])            
                if ((marker == "RKnee" and markersInfo[marker]["parent"] == "tibia_r") or  
                    (marker == "LKnee" and markersInfo[marker]["parent"] == "tibia_l") or
                    (marker == "RKnee" and markersInfo[marker]["parent"] == "sagittal_articulation_frame_r") or
                    (marker == "LKnee" and markersInfo[marker]["parent"] == "sagittal_articulation_frame_l") or
                    (marker == "RWrist" and markersInfo[marker]["parent"] == "hand_r") or
                    (marker == "LWrist" and markersInfo[marker]["parent"] == "hand_l")):
                    frame = joint.get_frames(1)
                else:
                    frame = joint.get_frames(0)
                assert frame.getName()[:-7] == markersInfo[marker]["parent"]
                location = frame.get_translation()
                
                if adjustLocationHipAnkle:
                    if marker == "RHip" or marker == "LHip":
                        
                        # Get scale factor based on saved factors from scaling.
                        body = bodySet.get(markersInfo[marker]["parent"])
                        attached_geometry = body.get_attached_geometry(0)
                        scale_factors = attached_geometry.get_scale_factors().to_numpy()
                        location_np = location.to_numpy()
                        
                        # After comparing triangulated OpenPose markers and
                        # mocap-based markers, it appears that the hip OpenPose
                        # markers should be located more forward and lateral.
                        location_adj_np = np.copy(location_np)
                        location_adj_np[0] += (hipOffsetX * scale_factors[0])
                        if marker == "RHip":
                            location_adj_np[2] += (hipOffsetZ * scale_factors[2])
                        elif marker == "LHip":
                            location_adj_np[2] -= (hipOffsetZ * scale_factors[2])
                        
                        location = opensim.Vec3(location_adj_np)
                
                parentFrame = "/bodyset/" + markersInfo[marker]["parent"]
