import glob
import os
import sys
import unittest
import numpy as np
import pickle

thisDir = os.path.dirname(os.path.realpath(__file__))
repoDir = os.path.abspath(os.path.join(thisDir,'../'))
sys.path.append(repoDir)
from utils import loadCameraParameters
from utilsSync import synchronizeVideos

import logging
logging.basicConfig(level=logging.DEBUG)

class SynchronizeVideos(unittest.TestCase):
    def testSynchronizeVideos(self):
        sessionName = 'sync-tests'
        dataDir = os.path.join(thisDir, 'opencap-test-data')
        
        trial_name_list = ['squats', 'walk', 'squats-with-arm-raise']
        sync_str_list = ['INFO:root:Using general sync function.',
                         'INFO:root:Using gait sync function.',
                         'INFO:root:Using handPunch sync function.']

        for trialName, sync_str in zip(trial_name_list, sync_str_list):
            with self.subTest(msg=f'subtest {trialName}',
                              sessionName=sessionName,
                              trialName=trialName,
                              dataDir=dataDir):
                
                sessionDir = os.path.join(dataDir, 'Data', sessionName)

                # original data insert Cam1 first, then Cam0 into the dictionary
                videosDir = os.path.join(dataDir, 'Data', 'sync-tests', 'Videos')
                cameraDirectories = {'Cam1': os.path.join(videosDir, 'Cam1'), 
                                     'Cam0': os.path.join(videosDir, 'Cam0')}
                
                trialRelativePath = os.path.join('InputMedia',
                                                trialName, 
                                                f'{trialName}.mov')
                
                CamParamDict = {}
                for camName in cameraDirectories:
                    camDir = cameraDirectories[camName]
                    CamParams = loadCameraParameters(
                            os.path.join(camDir, "cameraIntrinsicsExtrinsics.pickle"))
                    CamParamDict[camName] = CamParams.copy()

                poseDetectorDirectory = ''
                undistortPoints = True
                filtFreqs = {'gait':12, 'default':500} # hard-coded 'default' values from main()
                confidenceThreshold = 0.4
                imageBasedTracker = False
                camerasToUse_c = ['all']
                poseDetector = 'mmpose'
                resolutionPoseDetection = 'default'

                with self.assertLogs(level='INFO') as cm:
                    keypoints2D, confidence, keypointNames, frameRate, nansInOut, startEndFrames, cameras2Use = (
                        synchronizeVideos( 
                            cameraDirectories, trialRelativePath, poseDetectorDirectory,
                            undistortPoints=undistortPoints, CamParamDict=CamParamDict,
                            filtFreqs=filtFreqs, confidenceThreshold=confidenceThreshold,
                            imageBasedTracker=imageBasedTracker, cams2Use=camerasToUse_c, 
                            poseDetector=poseDetector, trialName=trialName,
                            resolutionPoseDetection=resolutionPoseDetection,
                            syncVer='1.1'))
                
                self.assertIn(sync_str, cm.output)
                
                ref_pkl = os.path.join(sessionDir, 'OutputReference', f'sync_{trialName}_output.pkl')
                with open(ref_pkl, 'rb') as f:
                    (ref_keypoints2D, ref_confidence, ref_keypointNames, ref_frameRate,
                        ref_nansInOut, ref_startEndFrames, ref_cameras2Use) = pickle.load(f)

                with self.subTest('keypoints2D'):
                    for dict_key in keypoints2D:
                        np.testing.assert_array_almost_equal(keypoints2D[dict_key],
                                                            ref_keypoints2D[dict_key])
                with self.subTest('confidence'):
                    for dict_key in confidence:
                        np.testing.assert_array_almost_equal(confidence[dict_key],
                                                            ref_confidence[dict_key])
                
                with self.subTest('keypointNames'):
                    self.assertEqual(keypointNames, ref_keypointNames)

                with self.subTest('frameRate'):
                    self.assertEqual(frameRate, ref_frameRate)

                with self.subTest('nansInOut'):
                    for dict_key in nansInOut:
                        np.testing.assert_array_almost_equal(nansInOut[dict_key],
                                                            ref_nansInOut[dict_key],
                                                            err_msg='nansInOut')
                with self.subTest('startEndFrames'):
                    for dict_key in startEndFrames:
                        np.testing.assert_array_almost_equal(startEndFrames[dict_key],
                                                            ref_startEndFrames[dict_key],
                                                            err_msg='startEndFrames')
                
                with self.subTest('cameras2Use'):
                    self.assertEqual(cameras2Use, ref_cameras2Use)


if __name__ == "__main__":
    unittest.main()
