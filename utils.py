import yaml
import json
import os
import socket
import requests
import urllib.request
import shutil
import utilsDataman
import pickle
import glob
import mimetypes
import subprocess
import zipfile

import numpy as np
import pandas as pd
from scipy import signal

from utilsAuth import getToken
from utilsAPI import getAPIURL

API_URL = getAPIURL()
API_TOKEN = getToken()

#%% Rest of utils

def getDataDirectory(isDocker=False):
    computername = socket.gethostname()
    # Paths to OpenPose folder for local testing.
    if computername == 'SUHLRICHHPLDESK':
        dataDir = 'C:/Users/scott.uhlrich/MyDrive/mobilecap/'
    elif computername == "LAPTOP-7EDI4Q8Q":
        dataDir = 'C:\MyDriveSym/mobilecap/'
    elif computername == 'DESKTOP-0UPR1OH':
        dataDir = 'C:/Users/antoi/Documents/MyRepositories/mobilecap_data/'
    elif computername == 'HPL1':
        dataDir = 'C:/Users/opencap/Documents/MyRepositories/mobilecap_data/'
    elif computername == 'DESKTOP-GUEOBL2':
        dataDir = 'C:/Users/opencap/Documents/MyRepositories/mobilecap_data/'
    elif computername == 'DESKTOP-L9OQ0MS':
        dataDir = 'C:/Users/antoi/Documents/MyRepositories/mobilecap_data/'
    elif computername == 'clarkadmin-MS-7996':
        dataDir = '/home/clarkadmin/Documents/MyRepositories/mobilecap_data/'
    elif computername == 'DESKTOP-NJMGEBG':
        dataDir = 'C:/Users/opencap/Documents/MyRepositories/mobilecap_data/'
    elif isDocker:
        dataDir = os.getcwd()
    else:
        dataDir = os.getcwd()
    return dataDir

def getOpenPoseDirectory(isDocker=False):
    computername = os.environ.get('COMPUTERNAME', None)
    
    # Paths to OpenPose folder for local testing.
    if computername == "DESKTOP-0UPR1OH":
        openPoseDirectory = "C:/Software/openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended/openpose"
    elif computername == "HPL1":
        openPoseDirectory = "C:/Users/opencap/Documents/MySoftware/openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended/openpose"
    elif computername == "DESKTOP-GUEOBL2":
        openPoseDirectory = "C:/Software/openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended/openpose"
    elif computername == "DESKTOP-L9OQ0MS":
        openPoseDirectory = "C:/Software/openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended/openpose"
    elif isDocker:
        openPoseDirectory = "docker"
    elif computername == 'SUHLRICHHPLDESK':
        openPoseDirectory = "C:/openpose/"
    elif computername == "LAPTOP-7EDI4Q8Q":
        openPoseDirectory = "C:/openpose/"
    elif computername == "DESKTOP-NJMGEBG":
        openPoseDirectory = "C:/openpose/"
    else:
        openPoseDirectory = "C:/openpose/"
    return openPoseDirectory

def getMMposeDirectory(isDocker=False):
    computername = socket.gethostname()
    
    # Paths to OpenPose folder for local testing.
    if computername == "clarkadmin-MS-7996":
        mmposeDirectory = "/home/clarkadmin/Documents/MyRepositories/MoVi_analysis/model_ckpts"
    else:
        mmposeDirectory = ''
    return mmposeDirectory

def loadCameraParameters(filename):
    open_file = open(filename, "rb")
    cameraParams = pickle.load(open_file)
    
    open_file.close()
    return cameraParams

def importMetadata(filePath):
    myYamlFile = open(filePath)
    parsedYamlFile = yaml.load(myYamlFile, Loader=yaml.FullLoader)
    
    return parsedYamlFile

def download_file(url, file_name):
    with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
        
def getTrialJson(trial_id):
    trialJson = requests.get(API_URL + "trials/{}/".format(trial_id),
                         headers = {"Authorization": "Token {}".format(API_TOKEN)}).json()
    return trialJson

def getSessionJson(session_id):
    sessionJson = requests.get(API_URL + "sessions/{}/".format(session_id),
                       headers = {"Authorization": "Token {}".format(API_TOKEN)}).json()
    
    # sort trials by time recorded
    def getCreatedAt(trial):
        return trial['created_at']
    sessionJson['trials'].sort(key=getCreatedAt)
    
    return sessionJson
    
def getTrialName(trial_id):
    trial = getTrialJson(trial_id)
    trial_name = trial['name']
    trial_name = trial_name.replace(' ', '')
    
    return trial_name

def writeMediaToAPI(API_URL,media_path,trial_id,tag=None,deleteOldMedia=False):
    
    if deleteOldMedia:
        deleteResult(trial_id, tag=tag)
    
    for filename in os.listdir(media_path):
        thisMimeType = mimetypes.guess_type(filename)
        if thisMimeType[0] is not None and not os.path.isdir(filename):
            print(filename)
            fileType = thisMimeType[0][0:thisMimeType[0].find('/')]
            if fileType == 'image' or fileType == 'video' or fileType == 'application':
                fullpath = "{}/{}".format(media_path, filename)
                
                if fileType == 'image' and tag == "calibration-img":
                    cam = filename[filename.find('Cam'):filename.find('.')]
                    if "altSoln" in filename:
                        altSoln = '_altSoln'
                    else:
                        altSoln = ''
                    device_id = cam + altSoln
                
                else:
                    device_id = None
                
                files = {'media': open(fullpath, 'rb')}
                data = {
                    "trial": trial_id,
                    "tag": tag,
                    "device_id" : device_id
                }
        
                r= requests.post("{}{}".format(API_URL, "results/"), files=files, data=data,
                         headers = {"Authorization": "Token {}".format(API_TOKEN)})
                files["media"].close()
                
                if r.status_code != 201:
                    print('server response was + ' + str(r.status_code))
                else:
                    print('Media results sent to API')
            
        
    return


def getTrialNameIdMapping(session_id):
    trials = getSessionJson(session_id)['trials']
    
    # dict of session name->id and date
    trialDict = {}
    for t in trials:
        trialDict[t['name']] = {'id':t['id'],'date':t['created_at']}
        
    return trialDict


def postCalibrationOptions(session_path,session_id,overwrite=False):
    calibration_id = getCalibrationTrialID(session_id)
    trial = getTrialJson(calibration_id)
   
    if trial['meta'] is None or overwrite == True:
        calibOptionsJsonPath = os.path.join(session_path,'Videos','calibOptionSelections.json')
        f = open(calibOptionsJsonPath)
        calibOptionsJson = json.load(f)
        f.close()
        data = {
                "meta":json.dumps({'calibration':calibOptionsJson})
            }
        trial_url = "{}{}{}/".format(API_URL, "trials/", calibration_id)
        r= requests.patch(trial_url, data=data,
              headers = {"Authorization": "Token {}".format(API_TOKEN)})
        
        if r.status_code == 200:
            print('Wrote calibration selections to metadata.')

def downloadVideosFromServer(session_id,trial_id, isDocker=True,
                             isCalibration=False, isStaticPose=False,
                             trial_name=None, session_name=None, 
                             session_path=None, benchmark=False):
    
    if session_name is None:
        session_name = session_id
    data_dir = getDataDirectory(isDocker)   
    if session_path is None:
        session_path = os.path.join(data_dir,'Data', session_name)  
    if not os.path.exists(session_path): 
        os.makedirs(session_path, exist_ok=True)
    
    trial = getTrialJson(trial_id)

    if trial_name is None:
        trial_name = trial['name']
    trial_name = trial_name.replace(' ', '')

    
    print("\nProcessing {}".format(trial_name))

    # The videos are not always organized in the same order. Here, we save
    # the order during the first trial processed in the session such that we
    # can use the same order for the other trials.
    if not benchmark:
        if not os.path.exists(os.path.join(session_path, "Videos", 'mappingCamDevice.pickle')):
            mappingCamDevice = {}
            for k, video in enumerate(trial["videos"]):
                os.makedirs(os.path.join(session_path, "Videos", "Cam{}".format(k), "InputMedia", trial_name), exist_ok=True)
                video_path = os.path.join(session_path, "Videos", "Cam{}".format(k), "InputMedia", trial_name, trial_id + ".mov")
                download_file(video["video"], video_path)                
                mappingCamDevice[video["device_id"].replace('-', '').upper()] = k
            with open(os.path.join(session_path, "Videos", 'mappingCamDevice.pickle'), 'wb') as handle:
                pickle.dump(mappingCamDevice, handle)
        else:
            with open(os.path.join(session_path, "Videos", 'mappingCamDevice.pickle'), 'rb') as handle:
                mappingCamDevice = pickle.load(handle)            
            for video in trial["videos"]:            
                k = mappingCamDevice[video["device_id"].replace('-', '').upper()] 
                videoDir = os.path.join(session_path, "Videos", "Cam{}".format(k), "InputMedia", trial_name)
                os.makedirs(videoDir, exist_ok=True)
                video_path = os.path.join(videoDir, trial_id + ".mov")
                if not os.path.exists(video_path):
                    if video['video'] :
                        download_file(video["video"], video_path)
    
        # Import and save metadata
        sessionYamlPath = os.path.join(session_path, "sessionMetadata.yaml")
        if not os.path.exists(sessionYamlPath) or isStaticPose or isCalibration:
            if isCalibration: # subject parameters won't be entered yet
                session_desc = getMetadataFromServer(session_id,justCheckerParams = isCalibration)
            else: # subject parameters will be entered when capturing static pose
                session_desc = getMetadataFromServer(session_id)      
                
            # Load iPhone models.
            phoneModel= []
            for i,video in enumerate(trial["videos"]):    
                phoneModel.append(video['parameters']['model'])
            session_desc['iphoneModel'] = {'Cam' + str(i) : phoneModel[i] for i in range(len(phoneModel))}
        
            # Save metadata.
            with open(sessionYamlPath, 'w') as file:
                yaml.dump(session_desc, file)
                
    return trial_name


def deleteCalibrationFiles(session_path):
    calImagePath = os.path.join(session_path,'CalibrationImages')
    if os.path.exists(calImagePath):
        shutil.rmtree(calImagePath)
    
    # Delete camera directories
    camDirs = glob.glob(os.path.join(session_path,'Videos','Cam*'))
    
    # Find extrinsic Filename
    extrinsicFileFound = False
    if len(camDirs)>1 and os.path.exists(os.path.join(camDirs[0], 'InputMedia')):
        inputDir = os.path.join(camDirs[0], 'InputMedia')
        dirContents = os.listdir(inputDir)
        trialNames = [tName for tName in dirContents if os.path.isdir(os.path.join(inputDir,tName))]
        for tName in trialNames:
            if os.path.exists(os.path.join(inputDir,tName,'extrinsicImage0.png')):
                extrinsicTrialName = tName
                extrinsicFileFound = True
    
    for camDir in camDirs:
        extPath = os.path.join(camDir,'cameraIntrinsicsExtrinsics.pickle')
        if os.path.exists(extPath):
            os.remove(extPath)
        #Find extrinsic Filename
        if extrinsicFileFound:
            extFolder = os.path.join(camDir,'InputMedia',extrinsicTrialName)
            if os.path.isdir(extFolder):
                shutil.rmtree(extFolder)
                
def deleteStaticFiles(session_path,staticTrialName='neutral'):
        
    vidDir = os.path.join(session_path,'Videos')
    camDirs = glob.glob(os.path.join(vidDir,'Cam*'))
    markerDirs = glob.glob(os.path.join(session_path,'MarkerData'))
    openSimDir = os.path.join(session_path,'OpenSimData')
    
    # This is a hack, but os.walk doesn't work on attached server drives
    for camDir in camDirs:
        mediaDirs = glob.glob(os.path.join(camDir,'*'))
        for medDir in mediaDirs:
            try:
                shutil.rmtree(os.path.join(camDir,medDir,staticTrialName))
                _,camName = os.path.split()
                print('deleting ' + camName + '/' + medDir + '/' + staticTrialName)
            except:
                pass
            
    for mkrDir in markerDirs:
        mkrFiles = glob.glob(os.path.join(mkrDir,'*'))
        for mkrFile in mkrFiles:
            if staticTrialName in mkrFile:
                os.remove(mkrFile)
                _,fName = os.split(mkrFile)
                print('deleting '+ fName)
           
    if os.path.exists(openSimDir):
        shutil.rmtree(openSimDir) # Static will be the first opensim data saved, so this is safe
        print('deleting openSimDir')
    
    # # this works locally, but not on server drives. Saving in case we change storage
    # for root, dirList, fileList in os.walk(session_path + './'):
    #     for thisFile in fileList:
    #         print(thisFile)
    #         if (bool(regex.match(staticTrialName + '.trc', thisFile)) or 
    #             bool(regex.match(staticTrialName + '.mot', thisFile)) or 
    #             bool(regex.match(staticTrialName + '.sto', thisFile))):
    #             filePath = os.path.join(root,thisFile)
    #             os.remove(filePath)
    #             print('removing ' + thisFile)
        
    #     for thisDir in dirList:
    #         print(thisDir)
    #         if thisDir == staticTrialName:
    #             dirPath = os.path.join(root,thisDir)
    #             shutil.rmtree(dirPath)
    #             print('removing ' + thisDir)

def switchCalibrationForCamera(cam,trial_id,session_path):
    trialName = getTrialName(trial_id)
    camPath = os.path.join(session_path,'Videos',cam)
    trialPath = os.path.join(camPath,'InputMedia',trialName)
    
    # change Picture 
    src = os.path.join(trialPath,'extrinsicCalib_soln1.jpg')
    dest = os.path.join(session_path,'CalibrationImages','extrinsicCalib' + cam + '.jpg')
    if os.path.exists(dest):
        os.remove(dest)
    shutil.copyfile(src,dest)
    
    # change calibration parameters
    src = os.path.join(trialPath,'cameraIntrinsicsExtrinsics_soln1.pickle')
    dest = os.path.join(camPath,'cameraIntrinsicsExtrinsics.pickle')
    if os.path.exists(dest):
        os.remove(dest)
    shutil.copyfile(src,dest)    
    
                 
def getMetadataFromServer(session_id,justCheckerParams=False):
    
    defaultMetadataPath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       'defaultSessionMetadata.yaml')
    session_desc = importMetadata(defaultMetadataPath)
    
    # Get session-specific metadata from api.

    session = getSessionJson(session_id)
    if session['meta'] is not None:
        if not justCheckerParams:
            session_desc["subjectID"] = session['meta']['subject']['id']
            session_desc["mass_kg"] = float(session['meta']['subject']['mass'])
            session_desc["height_m"] = float(session['meta']['subject']['height'])
            try:
                session_desc["posemodel"] = session['meta']['subject']['posemodel']
            except:
                session_desc["posemodel"] = 'openpose'
        if 'sessionWithCalibration' in session['meta'] and 'checkerboard' not in session['meta']:
            newSessionId = session['meta']['sessionWithCalibration']['id']
            session = getSessionJson(newSessionId)
                                   
        session_desc['checkerBoard']["squareSideLength_mm"] =  float(session['meta']['checkerboard']['square_size'])
        session_desc['checkerBoard']["black2BlackCornersWidth_n"] = int(session['meta']['checkerboard']['cols'])
        session_desc['checkerBoard']["black2BlackCornersHeight_n"] = int(session['meta']['checkerboard']['rows'])
        session_desc['checkerBoard']["placement"] = session['meta']['checkerboard']['placement']   
        

          
    else:
        print('Couldn''t find session metadata in API, using default metadata. May be issues.')
    
    return session_desc

def deleteResult(trial_id, tag=None,resultNum=None):
    # Delete specific result number, or all results with a specific tag, or all results if tag==None
    if resultNum != None:
        resultNums = [resultNum]
    elif tag != None:
        trial = getTrialJson(trial_id)
        resultNums = [r['id'] for r in trial['results'] if r['tag']==tag]
        
    elif tag == None: 
        trial = getTrialJson(trial_id)
        resultNums = [r['id'] for r in trial['results']]

    for rNum in resultNums:
        requests.delete(API_URL + "results/{}/".format(rNum),
                        headers = {"Authorization": "Token {}".format(API_TOKEN)})
        
def deleteAllResults(session_id):

    session = getSessionJson(session_id)
    
    for trial in session['trials']:
        deleteResult(trial['id'])

def writeCalibrationOptionsToAPI(session_path,session_id,calibration_id=None,trialName = None):
    if calibration_id == None:
        calibration_id = getCalibrationTrialID(session_id)
    
    if trialName == None:
        trial = getTrialJson(calibration_id)
        trialName = trial['name']
    
    deleteResult(calibration_id, tag='camera_mapping')
    videoDir = os.path.join(session_path,'Videos')
    camDirs = glob.glob(os.path.join(videoDir,'Cam*'))
    mapPath = os.path.join(videoDir, 'mappingCamDevice.pickle')
    postFileToTrial(mapPath,calibration_id,'camera_mapping','all')
    
    tag = 'calibration_parameters_options'
    deleteResult(calibration_id, tag=tag)
    for camDir in camDirs:
        _,camName = os.path.split(camDir)
        calibDir = os.path.join(camDir,'InputMedia',trialName)        
        # Post both solutions
        for i in range(2):
            filePath = os.path.join(calibDir,'cameraIntrinsicsExtrinsics_soln{}.pickle'.format(i))
            device_id = camName+'_soln{}'.format(i)
            postFileToTrial(filePath,calibration_id,tag,device_id)    

def getCalibrationTrialID(session_id):
    session = getSessionJson(session_id)
    
    calib_ids = [t['id'] for t in session['trials'] if t['name'] == 'calibration']
                                                          
    if len(calib_ids)>0:
        calibID = calib_ids[-1]
    elif session['meta']['sessionWithCalibration']:
        calibID = getCalibrationTrialID(session['meta']['sessionWithCalibration']['id'])
    else:
        raise Exception('No calibration trial in session.')
    
    return calibID

def getNeutralTrialID(session_id):
    session = getSessionJson(session_id)
    
    neutral_ids = [t['id'] for t in session['trials'] if t['name'] == 'neutral']
    
    if len(neutral_ids)>0:
        neutralID = neutral_ids[-1]
    elif session['meta']['neutral_trial']:
        neutralID = session['meta']['neutral_trial']['id']
    else:
        raise Exception('No neutral trial in session.')
    
    return neutralID       

def postCalibration(session_id,session_path,calibTrialID=None):
    
    videoDir = os.path.join(session_path,'Videos')
    videoFolders = glob.glob(os.path.join(videoDir,'Cam*'))
        
    if calibTrialID == None:
        calibTrialID = getCalibrationTrialID(session_id)
    
    # remove 'calibration_parameters' in case they exist already.
    tag = 'calibration_parameters'
    deleteResult(calibTrialID, tag=tag)
    for vf in videoFolders:
        _, camName = os.path.split(vf)
        fPath = os.path.join(vf,'cameraIntrinsicsExtrinsics.pickle')
        deviceID = camName
        postFileToTrial(fPath,calibTrialID,'calibration_parameters',deviceID)
    
    return

def getCalibration(session_id,session_path,trial_type='dynamic',getCalibrationOptions=False):
    # look for calibration pickles on Django. If they are not there, then see if 
    # we need to do any switch calibration, then post the good calibration to django.
    calibration_id = getCalibrationTrialID(session_id)

    # Check if calibration has been posted to session
    trial = getTrialJson(calibration_id)
    calibResultTags = [res['tag'] for res in trial['results']]

    # download the mapping
    videoFolder = os.path.join(session_path,'Videos')
    os.makedirs(videoFolder, exist_ok=True)
    mapURL = trial['results'][calibResultTags.index('camera_mapping')]['media']
    mapLocalPath = os.path.join(videoFolder,'mappingCamDevice.pickle')
    download_file(mapURL,mapLocalPath)
    
    # download calibration parameters and switch if necessary.
    calibrationOptions = downloadAndSwitchCalibrationFromDjango(session_id,session_path,
                                                                calibTrialID=calibration_id,
                                                                getCalibrationOptions=getCalibrationOptions)
    
    # Post calibration if neutral trial. The posted parameters are no longer
    # used, but it is handy to know which ones were selected from both options.
    if trial_type == 'static':
        postCalibration(session_id,session_path,calibTrialID=calibration_id)   

    if getCalibrationOptions:
        return calibrationOptions                             

def downloadAndSwitchCalibrationFromDjango(session_id,session_path,calibTrialID = None,
                                           getCalibrationOptions=False):
    if calibTrialID == None:
        calibTrialID = getCalibrationTrialID(session_id)
    trial = getTrialJson(calibTrialID)
       
    calibURLs = {t['device_id']:t['media'] for t in trial['results'] if t['tag'] == 'calibration_parameters_options'}
    
    if 'meta' in trial.keys() and trial['meta'] is not None and 'calibration' in trial['meta'].keys():
        calibDict = trial['meta']['calibration']
    else:
        print('No metadata for camera switching. Using first solution.')
        calibDict = {'Cam'+str(i):0 for i in range(len(trial['videos']))}
        
    for cam,calibNum in calibDict.items():
        camDir = os.path.join(session_path,'Videos',cam)
        os.makedirs(camDir,exist_ok=True)
        file_name = os.path.join(camDir,'cameraIntrinsicsExtrinsics.pickle')
        if calibNum == 0:
            download_file(calibURLs[cam+'_soln0'], file_name)
            print('Downloading calibration for ' + cam)
        elif calibNum == 1:
            download_file(calibURLs[cam+'_soln1'], file_name)                  
            print('Downloading alternate calibration camera for ' + cam)
    
    # If static trial and we are automatically selecting a calibration
    if getCalibrationOptions:
        tempPath = os.path.join(session_path,'tempCalib.pickle')
        calibrationOptions = {}
        for cam in calibDict.keys():
            calibrationOptions[cam] = []
            download_file(calibURLs[cam+'_soln0'], tempPath)
            calibrationOptions[cam].append(loadCameraParameters(tempPath))
            os.remove(tempPath)
            download_file(calibURLs[cam+'_soln1'], tempPath)
            calibrationOptions[cam].append(loadCameraParameters(tempPath))
            os.remove(tempPath)            
    
        return calibrationOptions
    else:
        return None
    
def changeSessionMetadata(session_ids,newMetaDict):   
   
    for session_id in session_ids:
        session_url = "{}{}{}/".format(API_URL, "sessions/", session_id)
        
        # get metadata
        session = getSessionJson(session_id)
        existingMeta = session['meta']
        
        for key in existingMeta.keys():
            if key in newMetaDict.keys():
                existingMeta[key] = newMetaDict[key]
            if type(existingMeta[key]) is dict:
                for key2 in existingMeta[key].keys():
                    if key2 in newMetaDict.keys():
                        existingMeta[key][key2] = newMetaDict[key2]
        
        data = {
                "meta":json.dumps(existingMeta)
            }
        
        r= requests.patch(session_url, data=data,
              headers = {"Authorization": "Token {}".format(API_TOKEN)})
        
        if r.status_code !=200:
            print('Changing metadata failed.')
            
        # Also change this in the metadata yaml in neutral trial
        trial_id = getNeutralTrialID(session_id)
        trial = getTrialJson(trial_id)
        resultTags = [res['tag'] for res in trial['results']]
        
        metaPath = os.path.join(os.getcwd(),'sessionMetadata.yaml')
        yamlURL = trial['results'][resultTags.index('session_metadata')]['media']
        download_file(yamlURL,metaPath)
        
        metaYaml = importMetadata(metaPath)
        
        for key in metaYaml.keys():
            if key in newMetaDict.keys():
                metaYaml[key] = newMetaDict[key]
            if type(metaYaml[key]) is dict:
                for key2 in metaYaml[key].keys():
                    if key2 in newMetaDict.keys():
                        metaYaml[key][key2] = newMetaDict[key2] 
                        
        with open(metaPath, 'w') as file:
            yaml.dump(metaYaml, file)
            
        deleteResult(trial_id, tag='session_metadata')
        postFileToTrial(metaPath,trial_id,tag='session_metadata',device_id='all')
        os.remove(metaPath)
        
def makeSessionPublic(session_id,publicStatus=True):
    
    session_url = "{}{}{}/".format(API_URL, "sessions/", session_id)
    
    data = {
            "public":publicStatus
        }
        
    r= requests.patch(session_url, data=data,
          headers = {"Authorization": "Token {}".format(API_TOKEN)})
    
    if r.status_code == 200:
        print('Successfully made ' + session_id + ' public.')
    else:
        print('server resp was ' + str(r.status_code))
        
    return

        
def postMotionData(trial_id,session_path,trial_name=None,isNeutral=False):
    if trial_name == None:
        trial_name = getTrialJson(trial_id)['id']
        
    # post pose pickles
    # If we parallelize this, this will be redundant, and we will want to delete this posting of pickles
    deleteResult(trial_id, tag='pose_pickle')
    camDirs = glob.glob(os.path.join(session_path,'Videos','Cam*'))
    for camDir in camDirs:
        outputPklFolder = glob.glob(os.path.join(camDir,'OutputPkl*'))[0]
        pklPath = glob.glob(os.path.join(outputPklFolder,trial_name,'*.pkl'))[0]
        _,camName = os.path.split(camDir)
        postFileToTrial(pklPath,trial_id,tag='pose_pickle',device_id=camName)
        
    # post marker data
    deleteResult(trial_id, tag='marker_data')
    markerDir = os.path.join(session_path,'MarkerData','PostAugmentation')
    markerPath = os.path.join(markerDir,trial_id + '.trc')
    postFileToTrial(markerPath,trial_id,tag='marker_data',device_id='all')
    
    if isNeutral:
        # post model
        deleteResult(trial_id, tag='opensim_model')
        modelFolder = os.path.join(session_path,'OpenSimData','Model')
        modelPath = glob.glob(modelFolder + '/*_scaled.osim')[0]
        postFileToTrial(modelPath,trial_id,tag='opensim_model',device_id='all')
        
        # post metadata
        deleteResult(trial_id, tag='session_metadata')
        metadataPath = os.path.join(session_path,'sessionMetadata.yaml')
        postFileToTrial(metadataPath,trial_id,tag='session_metadata',device_id='all')
    else:
        # post ik data
        deleteResult(trial_id, tag='ik_results')
        ikPath = os.path.join(session_path,'OpenSimData','Kinematics',trial_id + '.mot')
        postFileToTrial(ikPath,trial_id,tag='ik_results',device_id='all')
    
    # post settings
    deleteResult(trial_id, tag='main_settings')
    mainSettingsPath = os.path.join(markerDir,'Settings_{}.yaml'.format(trial_id))
    postFileToTrial(mainSettingsPath,trial_id,tag='main_settings',device_id='all')
        
    return

def getMotionData(trial_id,session_path,simplePath=False):
    trial = getTrialJson(trial_id)
    trial_name = trial['name']
    resultTags = [res['tag'] for res in trial['results']]

    # get marker data
    if 'ik_results' in resultTags:
        markerFolder = os.path.join(session_path,'MarkerData','PostAugmentation',trial_name)
        if simplePath:
            markerFolder = os.path.join(session_path,'MarkerData')
        markerPath = os.path.join(markerFolder,trial_name + '.trc')
        os.makedirs(markerFolder, exist_ok=True)
        markerURL = trial['results'][resultTags.index('marker_data')]['media']
        download_file(markerURL,markerPath)
    
    # get IK data
    if 'ik_results' in resultTags:
        ikFolder = os.path.join(session_path,'OpenSimData','Kinematics')
        if simplePath:
            ikFolder = os.path.join(session_path,'OpenSimData','Kinematics')
        ikPath = os.path.join(ikFolder,trial_name + '.mot')
        os.makedirs(ikFolder, exist_ok=True)
        ikURL = trial['results'][resultTags.index('ik_results')]['media']
        download_file(ikURL,ikPath)
    
    # TODO will want to get pose pickles eventually, once those are processed elsewhere
        
    return
        
def getModelAndMetadata(session_id,session_path,simplePath=False):
    neutral_id = getNeutralTrialID(session_id)
    trial = getTrialJson(neutral_id)
    resultTags = [res['tag'] for res in trial['results']]
    
    # get metadata
    metadataPath = os.path.join(session_path,'sessionMetadata.yaml')
    if not os.path.exists(metadataPath) :
        metadataURL = trial['results'][resultTags.index('session_metadata')]['media']
        download_file(metadataURL, metadataPath)
    
    # get model if does not exist
    modelURL = trial['results'][resultTags.index('opensim_model')]['media']
    modelName = modelURL[modelURL.rfind('-')+1:modelURL.rfind('?')]
    modelFolder = os.path.join(session_path,'OpenSimData','Model')
    if simplePath:
       modelFolder = os.path.join(session_path,'OpenSimData','Model')
    modelPath = os.path.join(modelFolder,modelName)
    if not os.path.exists(modelPath):
        os.makedirs(modelFolder, exist_ok=True)
        download_file(modelURL, modelPath)
        
    return
    
def postFileToTrial(filePath,trial_id,tag,device_id):
    files = {'media': open(filePath, 'rb')}
    data = {
        "trial": trial_id,
        "tag": tag,
        "device_id" : device_id
    }

    requests.post(API_URL + "results/", files=files, data=data,
                  headers = {"Authorization": "Token {}".format(API_TOKEN)})
    files["media"].close()
    
    return

def getSyncdVideos(trial_id,session_path):
    trial = getTrialJson(trial_id)
    trial_name = trial['name']
    
    if trial['results']:
        for result in trial['results']:
            if result['tag'] == 'video-sync':
                url = result['media']
                cam,suff = os.path.splitext(url[url.rfind('_')+1:])
                lastIdx = suff.find('?') 
                if lastIdx >0:
                    suff = suff[:lastIdx]
                
                syncVideoPath = os.path.join(session_path,'Videos',cam,'InputMedia',trial_name,trial_name + '_sync' + suff)
                download_file(url,syncVideoPath)
        
def downloadAndZipSession(session_id,deleteFolderWhenZipped=True,isDocker=True,
                          writeToDjango=False,justDownload=False,data_dir=None,
                          useSubjectNameFolder=False):
    
    session = getSessionJson(session_id)
    
    if data_dir is None:
        data_dir = os.path.join(getDataDirectory(isDocker=isDocker),'Data')
    if useSubjectNameFolder:
        folderName = session['name']
    else:
        folderName = session_id
    session_path = os.path.join(data_dir,folderName)
    
    calib_id = getCalibrationTrialID(session_id)
    neutral_id = getNeutralTrialID(session_id)
    dynamic_ids = [t['id'] for t in session['trials'] if (t['name'] != 'calibration' and t['name'] !='neutral')]
       
    # Calibration
    downloadVideosFromServer(session_id,calib_id,isDocker=isDocker,
                         isCalibration=True,isStaticPose=False) 
    getCalibration(session_id,session_path)
    
    # Neutral
    getModelAndMetadata(session_id,session_path)
    getMotionData(neutral_id,session_path)
    downloadVideosFromServer(session_id,neutral_id,isDocker=isDocker,
                     isCalibration=False,isStaticPose=True,session_path=session_path)
    getSyncdVideos(neutral_id,session_path)

    # Dynamic
    for dynamic_id in dynamic_ids:
        getMotionData(dynamic_id,session_path)
        downloadVideosFromServer(session_id,dynamic_id,isDocker=isDocker,
                 isCalibration=False,isStaticPose=False,session_path=session_path)
        getSyncdVideos(dynamic_id,session_path)

   
    if not justDownload:
        # Zip   
        def zipdir(path, ziph):
            # ziph is zipfile handle
            for root, dirs, files in os.walk(path):
                for file in files:
                    ziph.write(os.path.join(root, file), 
                               os.path.relpath(os.path.join(root, file), 
                                               os.path.join(path, '..')))
        
        session_zip = '{}.zip'.format(session_path)
    
        if os.path.isfile(session_zip):
            os.remove(session_zip)
      
        zipf = zipfile.ZipFile(session_zip, 'w', zipfile.ZIP_DEFLATED)
        zipdir(session_path, zipf)
        zipf.close()
        
        # write zip as a result to last trial for now
        if writeToDjango:
            postFileToTrial(session_zip,dynamic_ids[-1],tag='session_zip',device_id='all')
        
        if deleteFolderWhenZipped:
            if os.path.exists(session_path):
                shutil.rmtree(session_path)
            if os.path.exists(session_zip):
                os.remove(session_zip)
    
    return
#test session
# downloadAndZipSession('a24a895a-aa62-4403-bd9e-cf637ac02eb6',deleteFolderWhenZipped=False,isDocker=False)


def numpy2TRC(f, data, headers, fc=50.0, t_start=0.0, units="m"):
    
    header_mapping = {}
    for count, header in enumerate(headers):
        header_mapping[count+1] = header 
        
    # Line 1.
    f.write('PathFileType  4\t(X/Y/Z) %s\n' % os.getcwd())
    
    # Line 2.
    f.write('DataRate\tCameraRate\tNumFrames\tNumMarkers\t'
                'Units\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n')
    
    num_frames=data.shape[0]
    num_markers=len(header_mapping.keys())
    
    # Line 3.
    f.write('%.1f\t%.1f\t%i\t%i\t%s\t%.1f\t%i\t%i\n' % (
            fc, fc, num_frames,
            num_markers, units, fc,
            1, num_frames))
    
    # Line 4.
    f.write("Frame#\tTime\t")
    for key in sorted(header_mapping.keys()):
        f.write("%s\t\t\t" % format(header_mapping[key]))

    # Line 5.
    f.write("\n\t\t")
    for imark in np.arange(num_markers) + 1:
        f.write('X%i\tY%s\tZ%s\t' % (imark, imark, imark))
    f.write('\n')
    
    # Line 6.
    f.write('\n')

    for frame in range(data.shape[0]):
        f.write("{}\t{:.8f}\t".format(frame+1,(frame)/fc+t_start)) # opensim frame labeling is 1 indexed

        for key in sorted(header_mapping.keys()):
            f.write("{:.5f}\t{:.5f}\t{:.5f}\t".format(data[frame,0+(key-1)*3], data[frame,1+(key-1)*3], data[frame,2+(key-1)*3]))
        f.write("\n")
        
def numpy2storage(labels, data, storage_file):
    
    assert data.shape[1] == len(labels), "# labels doesn't match columns"
    assert labels[0] == "time"
    
    f = open(storage_file, 'w')
    f.write('name %s\n' %storage_file)
    f.write('datacolumns %d\n' %data.shape[1])
    f.write('datarows %d\n' %data.shape[0])
    f.write('range %f %f\n' %(np.min(data[:, 0]), np.max(data[:, 0])))
    f.write('endheader \n')
    
    for i in range(len(labels)):
        f.write('%s\t' %labels[i])
    f.write('\n')
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            f.write('%20.8f\t' %data[i, j])
        f.write('\n')
        
    f.close() 
      
    
def lowpassFilter(inputData, filtFreq, order=4):
    # Input is an array of nSteps x (nMeasures +1) because time is the first column
    time = inputData[:,0]
    fs=1/np.mean(np.diff(time))
    wn = filtFreq/(fs/2)
    sos = signal.butter(order/2,wn,btype='low',output='sos')
    inputDataFilt = signal.sosfiltfilt(sos,inputData[:,1:],axis=0)    
    data = np.concatenate((np.expand_dims(time,1), inputDataFilt), axis=1)

    return data

        
def TRC2numpy(pathFile, markers,rotation=None):
    # rotation is a dict, eg. {'y':90} with axis, angle for rotation
    
    trc_file = utilsDataman.TRCFile(pathFile)
    time = trc_file.time
    num_frames = time.shape[0]
    data = np.zeros((num_frames, len(markers)*3))
    
    if rotation != None:
        for axis,angle in rotation.items():
            trc_file.rotate(axis,angle)
    for count, marker in enumerate(markers):
        data[:,3*count:3*count+3] = trc_file.marker(marker)    
    this_dat = np.empty((num_frames, 1))
    this_dat[:, 0] = time
    data_out = np.concatenate((this_dat, data), axis=1)
    
    return data_out

def getOpenPoseMarkerNames():
    
    markerNames = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist",
                   "LShoulder", "LElbow", "LWrist", "midHip", "RHip",
                   "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye",
                   "LEye", "REar", "LEar", "LBigToe", "LSmallToe",
                   "LHeel", "RBigToe", "RSmallToe", "RHeel"]
    
    return markerNames

def getOpenPoseFaceMarkers():
    
    faceMarkerNames = ['Nose', 'REye', 'LEye', 'REar', 'LEar']
    markerNames = getOpenPoseMarkerNames()
    idxFaceMarkers = [markerNames.index(i) for i in faceMarkerNames]
    
    return faceMarkerNames, idxFaceMarkers

def getMMposeMarkerNames():
    
    markerNames = ["Nose", "LEye", "REye", "LEar", "REar", "LShoulder", 
                   "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist",
                   "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle",
                   "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe",
                   "RHeel"]        
    
    return markerNames


def rewriteVideos(inputPath,startFrame,nFrames,frameRate,outputDir=None,
                  imageScaleFactor = .5,outputFileName=None):
        
    inputDir, vidName = os.path.split(inputPath)
    vidName, vidExt = os.path.splitext(vidName)

    if outputFileName is None:
        outputFileName = vidName + '_sync' + vidExt
    if outputDir is not None:
        outputFullPath = os.path.join(outputDir, outputFileName)
    else:
        outputFullPath = os.path.join(inputDir, outputFileName)
      
    imageScaleArg = '' # None if want to keep image size the same
    maintainQualityArg = '-acodec copy -vcodec copy'
    if imageScaleFactor is not None:
        imageScaleArg = '-vf scale=iw/{:.0f}:-1'.format(1/imageScaleFactor)
        maintainQualityArg = ''

    startTime = startFrame/frameRate

    # We need to replace double space to single space for split to work
    # That's a bit hacky but works for now. (TODO)
    ffmpegCmd = "ffmpeg -loglevel error -y -ss {:.3f} -i {} {} -vframes {:.0f} {} {}".format(
                startTime, inputPath, maintainQualityArg, 
                nFrames, imageScaleArg, outputFullPath).rstrip().replace("  ", " ")

    subprocess.run(ffmpegCmd.split(" "))
    
    return

# %%  Found here: https://github.com/chrisdembia/perimysium/ => thanks Chris
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
	
def getIK(storage_file, joints, degrees=False):
    # Extract data
    data = storage2numpy(storage_file)
    Qs = pd.DataFrame(data=data['time'], columns=['time'])    
    for count, joint in enumerate(joints):  
        if ((joint == 'pelvis_tx') or (joint == 'pelvis_ty') or 
            (joint == 'pelvis_tz')):
            Qs.insert(count + 1, joint, data[joint])         
        else:
            if degrees == True:
                Qs.insert(count + 1, joint, data[joint])                  
            else:
                Qs.insert(count + 1, joint, data[joint] * np.pi / 180)              
            
    # Filter data    
    fs=1/np.mean(np.diff(Qs['time']))    
    fc = 6  # Cut-off frequency of the filter
    order = 4
    w = fc / (fs / 2) # Normalize the frequency
    b, a = signal.butter(order/2, w, 'low')  
    output = signal.filtfilt(b, a, Qs.loc[:, Qs.columns != 'time'], axis=0, 
                             padtype='odd', padlen=3*(max(len(b),len(a))-1))    
    output = pd.DataFrame(data=output, columns=joints)
    QsFilt = pd.concat([pd.DataFrame(data=data['time'], columns=['time']), 
                        output], axis=1)    
    
    return Qs, QsFilt

# %% Markers for augmenters.
def getOpenPoseMarkers_fullBody():

    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RKnee", "LKnee",
        "RAnkle", "LAnkle", "RHeel", "LHeel", "RSmallToe", "LSmallToe",
        "RBigToe", "LBigToe", "RElbow", "LElbow", "RWrist", "LWrist"]

    response_markers = ["C7_study", "r_shoulder_study", "L_shoulder_study",
                        "r.ASIS_study", "L.ASIS_study", "r.PSIS_study", 
                        "L.PSIS_study", "r_knee_study", "L_knee_study",
                        "r_mknee_study", "L_mknee_study", "r_ankle_study", 
                        "L_ankle_study", "r_mankle_study", "L_mankle_study",
                        "r_calc_study", "L_calc_study", "r_toe_study", 
                        "L_toe_study", "r_5meta_study", "L_5meta_study",
                        "r_lelbow_study", "L_lelbow_study", "r_melbow_study",
                        "L_melbow_study", "r_lwrist_study", "L_lwrist_study",
                        "r_mwrist_study", "L_mwrist_study",
                        "r_thigh1_study", "r_thigh2_study", "r_thigh3_study",
                        "L_thigh1_study", "L_thigh2_study", "L_thigh3_study", 
                        "r_sh1_study", "r_sh2_study", "r_sh3_study", 
                        "L_sh1_study", "L_sh2_study", "L_sh3_study",
                        "RHJC_study", "LHJC_study"]

    return feature_markers, response_markers

def getMMposeMarkers_fullBody():

    # Here we replace RSmallToe_mmpose and LSmallToe_mmpose by RSmallToe and
    # LSmallToe, since this is how they are named in the triangulation.
    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RKnee", "LKnee",
        "RAnkle", "LAnkle", "RHeel", "LHeel", "RSmallToe", "LSmallToe", 
        "RElbow", "LElbow", "RWrist", "LWrist"]

    response_markers = ["C7_study", "r_shoulder_study", "L_shoulder_study",
                        "r.ASIS_study", "L.ASIS_study", "r.PSIS_study", 
                        "L.PSIS_study", "r_knee_study", "L_knee_study",
                        "r_mknee_study", "L_mknee_study", "r_ankle_study", 
                        "L_ankle_study", "r_mankle_study", "L_mankle_study",
                        "r_calc_study", "L_calc_study", "r_toe_study", 
                        "L_toe_study", "r_5meta_study", "L_5meta_study",
                        "r_lelbow_study", "L_lelbow_study", "r_melbow_study",
                        "L_melbow_study", "r_lwrist_study", "L_lwrist_study",
                        "r_mwrist_study", "L_mwrist_study",
                        "r_thigh1_study", "r_thigh2_study", "r_thigh3_study",
                        "L_thigh1_study", "L_thigh2_study", "L_thigh3_study", 
                        "r_sh1_study", "r_sh2_study", "r_sh3_study", 
                        "L_sh1_study", "L_sh2_study", "L_sh3_study",
                        "RHJC_study", "LHJC_study"]

    return feature_markers, response_markers        

def getOpenPoseMarkers_lowerExtremity():

    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RKnee", "LKnee",
        "RAnkle", "LAnkle", "RHeel", "LHeel", "RSmallToe", "LSmallToe",
        "RBigToe", "LBigToe"]

    response_markers = ["C7_study", "r_shoulder_study", "L_shoulder_study",
                        "r.ASIS_study", "L.ASIS_study", "r.PSIS_study", 
                        "L.PSIS_study", "r_knee_study", "L_knee_study",
                        "r_mknee_study", "L_mknee_study", "r_ankle_study", 
                        "L_ankle_study", "r_mankle_study", "L_mankle_study",
                        "r_calc_study", "L_calc_study", "r_toe_study", 
                        "L_toe_study", "r_5meta_study", "L_5meta_study",
                        "r_thigh1_study", "r_thigh2_study", "r_thigh3_study",
                        "L_thigh1_study", "L_thigh2_study", "L_thigh3_study", 
                        "r_sh1_study", "r_sh2_study", "r_sh3_study", 
                        "L_sh1_study", "L_sh2_study", "L_sh3_study",
                        "RHJC_study", "LHJC_study"]

    return feature_markers, response_markers

def getMMposeMarkers_lowerExtremity():

    # Here we replace RSmallToe_mmpose and LSmallToe_mmpose by RSmallToe and
    # LSmallToe, since this is how they are named in the triangulation.
    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RKnee", "LKnee",
        "RAnkle", "LAnkle", "RHeel", "LHeel", "RSmallToe", "LSmallToe"]

    response_markers = ["C7_study", "r_shoulder_study", "L_shoulder_study",
                        "r.ASIS_study", "L.ASIS_study", "r.PSIS_study", 
                        "L.PSIS_study", "r_knee_study", "L_knee_study",
                        "r_mknee_study", "L_mknee_study", "r_ankle_study", 
                        "L_ankle_study", "r_mankle_study", "L_mankle_study",
                        "r_calc_study", "L_calc_study", "r_toe_study", 
                        "L_toe_study", "r_5meta_study", "L_5meta_study",
                        "r_thigh1_study", "r_thigh2_study", "r_thigh3_study",
                        "L_thigh1_study", "L_thigh2_study", "L_thigh3_study", 
                        "r_sh1_study", "r_sh2_study", "r_sh3_study", 
                        "L_sh1_study", "L_sh2_study", "L_sh3_study",
                        "RHJC_study", "LHJC_study"]

    return feature_markers, response_markers

def getMarkers_upperExtremity_pelvis():

    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RElbow", "LElbow",
        "RWrist", "LWrist"]

    response_markers = ["r_lelbow_study", "L_lelbow_study", "r_melbow_study",
                        "L_melbow_study", "r_lwrist_study", "L_lwrist_study",
                        "r_mwrist_study", "L_mwrist_study"]

    return feature_markers, response_markers

def getMarkers_upperExtremity_noPelvis():

    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RElbow", "LElbow", "RWrist",
        "LWrist"]

    response_markers = ["r_lelbow_study", "L_lelbow_study", "r_melbow_study",
                        "L_melbow_study", "r_lwrist_study", "L_lwrist_study",
                        "r_mwrist_study", "L_mwrist_study"]

    return feature_markers, response_markers

def delete_multiple_element(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)

def getVideoExtension(pathFileWithoutExtension):
    
    pathVideoDir = os.path.split(pathFileWithoutExtension)[0]
    videoName = os.path.split(pathFileWithoutExtension)[1]
    for file in os.listdir(pathVideoDir):
        if videoName == file.rsplit('.', 1)[0]:
            extension = '.' + file.rsplit('.', 1)[1]
            
    return extension