import glob
import os
import sys
import numpy as np
import pickle
import pytest

thisDir = os.path.dirname(os.path.realpath(__file__))
repoDir = os.path.abspath(os.path.join(thisDir,'../'))
sys.path.append(repoDir)
from utils import loadCameraParameters
from utilsSync import synchronizeVideos

import logging
logging.basicConfig(level=logging.DEBUG)

dataDir = os.path.join(thisDir, 'opencap-test-data')
sessionName = 'sync-tests'
sessionDir = os.path.join(dataDir, 'Data', sessionName)
videosDir = os.path.join(dataDir, 'Data', 'sync-tests', 'Videos')
cameraDirectories = {'Cam1': os.path.join(videosDir, 'Cam1'), 
                     'Cam0': os.path.join(videosDir, 'Cam0')}

trial_name_list = ['squats', 'walk', 'squats-with-arm-raise']
sync_str_list = ['Using general sync function.',
                 'Using gait sync function.',
                 'Using handPunch sync function.']
sync_ver_list = ['1.0', '1.1']

@pytest.mark.parametrize("sync_ver", sync_ver_list)
@pytest.mark.parametrize("trial_name,sync_str", zip(trial_name_list, sync_str_list))
def test_synchronize_videos(trial_name, sync_str, sync_ver, caplog):
    caplog.set_level(logging.INFO)
    trialRelativePath = os.path.join('InputMedia', trial_name, f'{trial_name}.mov')
    CamParamDict = {}
    for camName in cameraDirectories:
        camDir = cameraDirectories[camName]
        CamParams = loadCameraParameters(os.path.join(camDir, "cameraIntrinsicsExtrinsics.pickle"))
        CamParamDict[camName] = CamParams.copy()

    poseDetectorDirectory = ''
    undistortPoints = True
    filtFreqs = {'gait':12, 'default':500}
    confidenceThreshold = 0.4
    imageBasedTracker = False
    camerasToUse_c = ['all']
    poseDetector = 'mmpose'
    resolutionPoseDetection = 'default'

    keypoints2D, confidence, keypointNames, frameRate, nansInOut, startEndFrames, cameras2Use = (
        synchronizeVideos( 
            cameraDirectories, trialRelativePath, poseDetectorDirectory,
            undistortPoints=undistortPoints, CamParamDict=CamParamDict,
            filtFreqs=filtFreqs, confidenceThreshold=confidenceThreshold,
            imageBasedTracker=imageBasedTracker, cams2Use=camerasToUse_c, 
            poseDetector=poseDetector, trialName=trial_name,
            resolutionPoseDetection=resolutionPoseDetection,
            syncVer=sync_ver))

    assert sync_str in caplog.text

    ref_pkl = os.path.join(sessionDir, 'OutputReference', f'sync_{trial_name}_output.pkl')
    with open(ref_pkl, 'rb') as f:
        (ref_keypoints2D, ref_confidence, ref_keypointNames, ref_frameRate,
            ref_nansInOut, ref_startEndFrames, ref_cameras2Use) = pickle.load(f)
    
    # Check that keypoints2D, confidence, nansInOut, and startEndFrames
    # share the same keys as each other
    assert set(keypoints2D.keys()) == set(confidence.keys()) == set(nansInOut.keys()) == set(startEndFrames.keys())

    # For these data, sync 1.1 shifts sync lag by 1 frame later for 
    # hand punch test (4 frames instead of 3 frames), so we adjust reference 
    # data (startEndFrames, keypoints, and confidence) accordingly for each 
    # camera.
    if sync_ver == '1.1' and trial_name == 'squats-with-arm-raise':
        ref_startEndFrames = {'Cam1': [4, 798], 'Cam0': [0, 794]}
        start_end_frames_from_ref = {'Cam1': [1, 795], 'Cam0': [0, 794]}
        for dict_key in ref_keypoints2D:
            start_frame = start_end_frames_from_ref[dict_key][0]
            end_frame = start_end_frames_from_ref[dict_key][1] + 1 #include end frame

            ref_keypoints2D[dict_key] = ref_keypoints2D[dict_key][:,start_frame:end_frame,:]
            ref_confidence[dict_key] = ref_confidence[dict_key][:,start_frame:end_frame]         

    for dict_key in keypoints2D:
        np.testing.assert_array_almost_equal(keypoints2D[dict_key], ref_keypoints2D[dict_key], err_msg=f'keypoints2D: {dict_key}')
    for dict_key in confidence:
        np.testing.assert_array_almost_equal(confidence[dict_key], ref_confidence[dict_key], err_msg=f'confidence: {dict_key}')
    assert keypointNames == ref_keypointNames
    assert frameRate == ref_frameRate
    for dict_key in nansInOut:
        np.testing.assert_array_almost_equal(nansInOut[dict_key], ref_nansInOut[dict_key], err_msg=f'nansInOut: {dict_key}')
    for dict_key in startEndFrames:
        np.testing.assert_array_almost_equal(startEndFrames[dict_key], ref_startEndFrames[dict_key], err_msg=f'startEndFrames: {dict_key}')
    assert cameras2Use == ref_cameras2Use
