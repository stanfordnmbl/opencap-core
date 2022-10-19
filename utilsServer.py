import os
import glob
import shutil
import requests
import json

from main import main
from utils import getDataDirectory
from utils import getTrialName
from utils import downloadVideosFromServer
from utils import switchCalibrationForCamera
from utils import deleteCalibrationFiles
from utils import deleteStaticFiles
from utils import writeMediaToAPI
from utils import postCalibrationOptions
from utils import getCalibration
from utils import getModelAndMetadata
from utils import writeCalibrationOptionsToAPI
from utils import postMotionData
from utils import getNeutralTrialID
from utils import  getCalibrationTrialID
from utilsAuth import getToken
from utilsAPI import getAPIURL


API_URL = getAPIURL()
API_TOKEN = getToken()

def processTrial(session_id, trial_id, trial_type = 'dynamic',
                 imageUpsampleFactor = 4, poseDetector = 'OpenPose',
                 isDocker = True, resolutionPoseDetection='default',
                 extrinsicTrialName = 'calibration',
                 deleteLocalFolder = True,
                 hasWritePermissions=True):

    # Get session directory
    session_name = session_id 
    data_dir = getDataDirectory(isDocker=isDocker)
    session_path = os.path.join(data_dir,'Data',session_name)    
    trial_url = "{}{}{}/".format(API_URL, "trials/", trial_id)
       
    # Process the 3 different types of trials
    if trial_type == 'calibration':
        # delete extrinsic files if they exist.
        deleteCalibrationFiles(session_path)
        
        # download the videos
        trial_name = downloadVideosFromServer(session_id,trial_id,isDocker=isDocker,
                                 isCalibration=True,isStaticPose=False)
        
        # run calibration
        try:
            main(session_name, trial_name, trial_id, isDocker=isDocker, extrinsicsTrial=True,
                 imageUpsampleFactor=imageUpsampleFactor,genericFolderNames = True,
                 poseDetector=poseDetector)
        except Exception as e:
            error_msg = {}
            error_msg['error_msg'] = e.args[0]
            error_msg['error_msg_dev'] = e.args[1]
            _ = requests.patch(trial_url, data={"meta": json.dumps(error_msg)},
                   headers = {"Authorization": "Token {}".format(API_TOKEN)})   
            raise Exception('Calibration failed')
        
        if not hasWritePermissions:
            print('You are not the owner of this session, so do not have permission to write results to database.')
            return
            
        # Write calibration images to django
        images_path = os.path.join(session_path,'CalibrationImages')
        writeMediaToAPI(API_URL,images_path,trial_id,tag="calibration-img",deleteOldMedia=True)
        
        # Write calibration solutions to django
        writeCalibrationOptionsToAPI(session_path,session_id,calibration_id = trial_id,
                                     trialName = extrinsicTrialName)
        return
        
    elif trial_type == 'static':
        # delete static files if they exist.
        deleteStaticFiles(session_path, staticTrialName = 'neutral')
        
        # Check for calibration to use on django, if not, check for switch calibrations and post result.
        calibrationOptions = getCalibration(session_id,session_path,trial_type=trial_type,getCalibrationOptions=True)   
        
        # download the videos
        trial_name = downloadVideosFromServer(session_id,trial_id,isDocker=isDocker,
                                 isCalibration=False,isStaticPose=True)
        
        # run static
        try:
            main(session_name, trial_name, trial_id, isDocker=isDocker, extrinsicsTrial=False,
                 poseDetector=poseDetector,
                 imageUpsampleFactor=imageUpsampleFactor,
                 scaleModel = True,
                 resolutionPoseDetection = resolutionPoseDetection,
                 genericFolderNames = True,
                 calibrationOptions = calibrationOptions)
        except Exception as e:            
            error_msg = {}
            error_msg['error_msg'] = e.args[0]
            error_msg['error_msg_dev'] = e.args[1]
            _ = requests.patch(trial_url, data={"meta": json.dumps(error_msg)},
                   headers = {"Authorization": "Token {}".format(API_TOKEN)})
            raise Exception('Static trial failed')
        
        if not hasWritePermissions:
            print('You are not the owner of this session, so do not have permission to write results to database.')
            return
        
        # Write videos to django
        video_path = getResultsPath(session_id, trial_id,
                                    resultType='neutralVideo', isDocker=isDocker)
        writeMediaToAPI(API_URL,video_path,trial_id, tag='video-sync',deleteOldMedia=True)
        
        # Write neutral pose images to django
        images_path = os.path.join(session_path,'NeutralPoseImages')
        writeMediaToAPI(API_URL,images_path,trial_id,tag="neutral-img",deleteOldMedia=True)
        
        # Write visualizer jsons to django
        visualizerJson_path = getResultsPath(session_id, trial_id, 
                                             resultType='visualizerJson', 
                                             isDocker=isDocker)
        writeMediaToAPI(API_URL,visualizerJson_path,trial_id,
                        tag="visualizerTransforms-json",deleteOldMedia=True)
        
        # Write results to django
        postMotionData(trial_id,session_path,trial_name=trial_name,isNeutral=True)
        
        # Write calibration options to django
        postCalibrationOptions(session_path,session_id,overwrite=True)
        
    elif trial_type == 'dynamic':
        # download calibration, model, and metadata if not existing
        getCalibration(session_id,session_path,trial_type=trial_type)   
        getModelAndMetadata(session_id,session_path)
        
        # download the videos
        trial_name = downloadVideosFromServer(
            session_id, trial_id, isDocker=isDocker, isCalibration=False,
            isStaticPose=False)
        
        # run dynamic
        try:
            main(session_name, trial_name, trial_id, isDocker=isDocker, extrinsicsTrial=False,
                 poseDetector=poseDetector,
                 imageUpsampleFactor=imageUpsampleFactor,
                 resolutionPoseDetection = resolutionPoseDetection,
                 genericFolderNames = True)
        except Exception as e:
            error_msg = {}
            error_msg['error_msg'] = e.args[0]
            error_msg['error_msg_dev'] = e.args[1]
            _ = requests.patch(trial_url, data={"meta": json.dumps(error_msg)},
                   headers = {"Authorization": "Token {}".format(API_TOKEN)})   
            raise Exception('Dynamic trial failed.\n' + error_msg['error_msg_dev'])
        
        if not hasWritePermissions:
            print('You are not the owner of this session, so do not have permission to write results to database.')
            return
        
        # Write videos to django
        video_path = getResultsPath(session_id, trial_id,
                                    resultType='sync_video', isDocker=isDocker)
        writeMediaToAPI(API_URL,video_path,trial_id, tag='video-sync',deleteOldMedia=True)
        
        # Write visualizer jsons to django
        visualizerJson_path = getResultsPath(session_id, trial_id, 
                                             resultType='visualizerJson', 
                                             isDocker=isDocker)
        writeMediaToAPI(API_URL,visualizerJson_path,trial_id,
                        tag="visualizerTransforms-json",deleteOldMedia=True)
        
        # Write results to django
        postMotionData(trial_id,session_path,trial_name=trial_name,isNeutral=False)
        
    else:
        raise Exception('Wrong trial type. Options: calibration, static, dynamic.')
        
    # Remove data

    if deleteLocalFolder:
        shutil.rmtree(session_path)
      
        
def getCalibrationImagePath(session_id,isDocker=True):
    session_name = session_id # TODO We may want to name this on server side?
    data_dir = getDataDirectory(isDocker=isDocker)
    session_path = os.path.join(data_dir,'Data',session_name)
    
    image_path = os.path.join(session_path,'CalibrationImages')
    
    return image_path

def getResultsPath(session_id, trial_id, resultType='sync_video', isDocker=True):
    session_name = session_id 
    data_dir = getDataDirectory(isDocker=isDocker)
    session_path = os.path.join(data_dir,'Data',session_name)  
    trial_name = getTrialName(trial_id)
    
    if resultType == 'sync_video':
        result_path = os.path.join(session_path, 'VisualizerVideos', trial_name)
    elif resultType == 'visualizerJson':
        result_path = os.path.join(session_path, 'VisualizerJsons', trial_name)
    elif resultType == 'neutralVideo':
        result_path = os.path.join(session_path, 'NeutralPoseImages','Videos')
    else:
        raise Exception('Wrong result type.')
    
    return result_path


def switchCalibration(session_id,trial_id,camList,isDocker=True):  
    # Get session directory
    session_name = session_id # TODO We may want to name this on server side?
    data_dir = getDataDirectory(isDocker=isDocker)
    session_path = os.path.join(data_dir,'Data',session_name)
    
    for cam in camList:
        switchCalibrationForCamera(cam,trial_id,session_path)
    

def newSessionSameSetup(session_id_old,session_id_new,extrinsicTrialName='calibration',isDocker=True):
    # Get session directory
    session_name_new = session_id_new # TODO We may want to name this on server side?
    session_name_old = session_id_old
    data_dir = getDataDirectory(isDocker=isDocker)
    session_path_new = os.path.join(data_dir,'Data',session_name_new)
    session_path_old = os.path.join(data_dir,'Data',session_name_old)
    
    os.makedirs(session_path_new, exist_ok=True)
    
    # Get cameras
    camList = [os.path.split(name)[1] for name in glob.glob(os.path.join(session_path_old,'Videos','Cam*'))]
    for cam in camList:
        # Calibration trial
        newCalibFolder = os.path.join(session_path_new,'Videos',cam,'InputMedia',extrinsicTrialName)
        shutil.copytree(os.path.join(session_path_old,'Videos',cam,'InputMedia',extrinsicTrialName),
                        newCalibFolder)
        # Calibration parameters
        shutil.copyfile(os.path.join(session_path_old,'Videos',cam,'cameraIntrinsicsExtrinsics.pickle'),
                        os.path.join(session_path_new,'Videos',cam,'cameraIntrinsicsExtrinsics.pickle'))
           
    # Copy metadata and CalibrationImages
    shutil.copytree(os.path.join(session_path_old,'CalibrationImages'),
                     os.path.join(session_path_new,'CalibrationImages'))
    shutil.copyfile(os.path.join(session_path_old,'sessionMetadata.yaml'),
                     os.path.join(session_path_new,'sessionMetadata.yaml'))
            
    
def batchReprocess(session_ids,calib_id,static_id,dynamic_ids,poseDetector='OpenPose', 
                   resolutionPoseDetection='default',deleteLocalFolder=True,
                   isServer=False):

    if (type(calib_id) == str or type(static_id) == str or type(dynamic_ids) == str or 
        (type(dynamic_ids)==list and len(dynamic_ids)>0)) and len(session_ids) >1:
        raise Exception('can only have one session number if hardcoding other trial ids')
        
    for session_id in session_ids:
        print('Processing ' + session_id)
        
        # check if write permissions (session owner or admin)
        permissions = requests.get(API_URL + "sessions/{}/get_session_permission/".format(session_id),
                     headers = {"Authorization": "Token {}".format(API_TOKEN)}).json()
        hasWritePermissions = permissions['isAdmin'] or permissions['isOwner']


        if calib_id == None:
            calib_id_toProcess = getCalibrationTrialID(session_id)
        else:
            calib_id_toProcess = calib_id
        
        if len(calib_id_toProcess) > 0:
            processTrial(session_id,
                          calib_id_toProcess,
                          trial_type="calibration",
                          poseDetector = poseDetector,
                          deleteLocalFolder = deleteLocalFolder,
                          isDocker=isServer,
                          hasWritePermissions = hasWritePermissions)
        
        if static_id == None:
            static_id_toProcess = getNeutralTrialID(session_id)
        else:
            static_id_toProcess = static_id
        
        if len(static_id_toProcess)  > 0:
            try:
                processTrial(session_id,
                              static_id_toProcess,
                              trial_type="static",
                              resolutionPoseDetection = resolutionPoseDetection,
                              poseDetector = poseDetector,
                              deleteLocalFolder = deleteLocalFolder,
                              isDocker=isServer,
                              hasWritePermissions = hasWritePermissions)
                statusData = {'status':'done'}
                _ = requests.patch("https://api.opencap.ai/trials/{}/".format(static_id_toProcess), data=statusData,
                         headers = {"Authorization": "Token {}".format(API_TOKEN)})
            except Exception as e:
                print(e)
                statusData = {'status':'error'}
                _ = requests.patch("https://api.opencap.ai/trials/{}/".format(static_id_toProcess), data=statusData,
                         headers = {"Authorization": "Token {}".format(API_TOKEN)})
        if dynamic_ids == None:

            session = requests.get("https://api.opencap.ai/sessions/{}/".format(session_id),
                                   headers = {"Authorization": "Token {}".format(API_TOKEN)}).json()
            dynamic_ids_toProcess = [t['id'] for t in session['trials'] if (t['name'] != 'calibration' and t['name'] !='neutral')]
        else:
            if type(dynamic_ids) == str:
                dynamic_ids_toProcess = [dynamic_ids]
            elif type(dynamic_ids) == list:
                dynamic_ids_toProcess=dynamic_ids
        
        for dID in dynamic_ids_toProcess:
            try:
                processTrial(session_id,
                          dID,
                          trial_type="dynamic",
                          resolutionPoseDetection = resolutionPoseDetection,
                          poseDetector = poseDetector,
                          deleteLocalFolder = deleteLocalFolder,
                          isDocker=isServer,
                          hasWritePermissions = hasWritePermissions)
                
                statusData = {'status':'done'}
                _ = requests.patch("https://api.opencap.ai/trials/{}/".format(dID), data=statusData,
                         headers = {"Authorization": "Token {}".format(API_TOKEN)})
            except Exception as e:
                print(e)
                statusData = {'status':'error'}
                _ = requests.patch("https://api.opencap.ai/trials/{}/".format(static_id_toProcess), data=statusData,
                         headers = {"Authorization": "Token {}".format(API_TOKEN)})
