import logging
import os
import sys
import unittest
import numpy as np
import pandas as pd

thisDir = os.path.dirname(os.path.realpath(__file__))
repoDir = os.path.abspath(os.path.join(thisDir,'../'))
sys.path.append(repoDir)
from main import main

def load_trc(file, num_metadata_lines=5):
    with open(file, 'r') as f:
        lines = f.readlines()
        metadata = lines[:num_metadata_lines]

    df = pd.read_csv(file, 
                     sep='\t', 
                     skiprows=num_metadata_lines+1, 
                     header=None)
    return df, metadata

def load_mot(file, num_metadata_lines=10):
    with open(file, 'r') as f:
        lines = f.readlines()
        metadata = lines[:num_metadata_lines]

    df = pd.read_csv(file, 
                     sep='\t', 
                     skiprows=num_metadata_lines)
    return df, metadata

def compare_mot(self, output_mot_df, ref_mot_df, t0, tf):
    # use t0 and tf to limit comparison to main part of motion
    output_mot_df_slice = output_mot_df[(output_mot_df['time'] >= t0) & 
                                        (output_mot_df['time'] <= tf)]
    ref_mot_df_slice = ref_mot_df[(ref_mot_df['time'] >= t0) & 
                                    (ref_mot_df['time'] <= tf)]

    for col in ref_mot_df.columns:
        print(f'Checking column {col}')

        # time column should be equal since IK is frame-by-frame
        if col == 'time':
            pd.testing.assert_series_equal(output_mot_df[col],
                                            ref_mot_df[col])
        
        # check translational within 1 mm max error
        # rmse within 0.2 mm
        elif any(substr in col for substr in ['tx', 'ty', 'tz']):
            pd.testing.assert_series_equal(output_mot_df_slice[col],
                                            ref_mot_df_slice[col],
                                            atol=0.002)
            
            rmse = calc_rmse(output_mot_df_slice[col], 
                                ref_mot_df_slice[col])
            self.assertLessEqual(rmse, 0.001)
        
        # check rotational within 2.5 degrees max error,
        # rmse within 0.5 degrees
        else:
            pd.testing.assert_series_equal(output_mot_df_slice[col],
                                        ref_mot_df_slice[col],
                                        atol=2.5)
            
            rmse = calc_rmse(output_mot_df_slice[col],
                                ref_mot_df_slice[col])
            self.assertLessEqual(rmse, 0.5)

def calc_rmse(series1, series2):
    return np.sqrt(((series1 - series2)**2).mean())



class SquatsWithArmRaise(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sessionName = 'sync-tests'
        trialName = 'squats-with-arm-raise'
        trialID = trialName
        cls.dataDir = os.path.join(thisDir, 'opencap-test-data')
        
        main(sessionName, trialName, trialID, 
            dataDir=cls.dataDir,
            genericFolderNames=True,
            poseDetector='hrnet')
        
    def testCompareMarkerData(self):
        # Compare marker data
        output_trc = os.path.join(self.dataDir, 'Data', 'sync-tests', 
                                  'MarkerData', 'PostAugmentation', 'squats-with-arm-raise.trc')
        ref_trc = os.path.join(self.dataDir, 'Data', 'sync-tests', 
                               'OutputReference', 'squats-with-arm-raise.trc')

        output_trc_df, output_trc_metadata = load_trc(output_trc)
        ref_trc_df, ref_trc_metadata = load_trc(ref_trc)

        pd.testing.assert_frame_equal(output_trc_df, ref_trc_df,
                                      check_exact=False,
                                      atol=1e-5)
    
    def testCompareIKData(self):
        # Compare kinematic data with different tolerances for rotational
        # and translational coordinates
        output_mot = os.path.join(self.dataDir, 'Data', 'sync-tests', 
                                  'OpenSimData', 'Kinematics', 'squats-with-arm-raise.mot')
        ref_mot = os.path.join(self.dataDir, 'Data', 'sync-tests', 
                               'OutputReference', 'squats-with-arm-raise.mot')

        output_mot_df, output_mot_metadata = load_mot(output_mot)
        ref_mot_df, ref_mot_metadata = load_mot(ref_mot)

        pd.testing.assert_index_equal(output_mot_df.columns, ref_mot_df.columns)

        # choose time points core to motion, doesn't include arm raise
        # that has more noise so test can be tighter
        t0 = 5.0
        tf = 10.0
        compare_mot(self, output_mot_df, ref_mot_df, t0, tf)

class SquatsNoArmRaise(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sessionName = 'sync-tests'
        trialName = 'squats'
        trialID = trialName
        cls.dataDir = os.path.join(thisDir, 'opencap-test-data')
        
        main(sessionName, trialName, trialID, 
            dataDir=cls.dataDir,
            genericFolderNames=True,
            poseDetector='hrnet')
        
    def testCompareMarkerData(self):
        # Compare marker data
        output_trc = os.path.join(self.dataDir, 'Data', 'sync-tests', 
                                  'MarkerData', 'PostAugmentation', 'squats.trc')
        ref_trc = os.path.join(self.dataDir, 'Data', 'sync-tests', 
                               'OutputReference', 'squats.trc')

        output_trc_df, output_trc_metadata = load_trc(output_trc)
        ref_trc_df, ref_trc_metadata = load_trc(ref_trc)

        pd.testing.assert_frame_equal(output_trc_df, ref_trc_df,
                                      check_exact=False,
                                      atol=1e-5)
    
    def testCompareIKData(self):
        # Compare kinematic data with different tolerances for rotational
        # and translational coordinates
        output_mot = os.path.join(self.dataDir, 'Data', 'sync-tests', 
                                  'OpenSimData', 'Kinematics', 'squats.mot')
        ref_mot = os.path.join(self.dataDir, 'Data', 'sync-tests', 
                               'OutputReference', 'squats.mot')

        output_mot_df, output_mot_metadata = load_mot(output_mot)
        ref_mot_df, ref_mot_metadata = load_mot(ref_mot)

        pd.testing.assert_index_equal(output_mot_df.columns, ref_mot_df.columns)

        t0 = 3.0
        tf = 8.0
        compare_mot(self, output_mot_df, ref_mot_df, t0, tf)

class WalkNoArmRaise(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sessionName = 'sync-tests'
        trialName = 'walk'
        trialID = trialName
        cls.dataDir = os.path.join(thisDir, 'opencap-test-data')
        
        main(sessionName, trialName, trialID, 
            dataDir=cls.dataDir,
            genericFolderNames=True,
            poseDetector='hrnet')
        
    def testCompareMarkerData(self):
        # Compare marker data
        output_trc = os.path.join(self.dataDir, 'Data', 'sync-tests', 
                                  'MarkerData', 'PostAugmentation', 'walk.trc')
        ref_trc = os.path.join(self.dataDir, 'Data', 'sync-tests', 
                               'OutputReference', 'walk.trc')

        output_trc_df, output_trc_metadata = load_trc(output_trc)
        ref_trc_df, ref_trc_metadata = load_trc(ref_trc)

        pd.testing.assert_frame_equal(output_trc_df, ref_trc_df,
                                      check_exact=False,
                                      atol=1e-5)
    
    def testCompareIKData(self):
        # Compare kinematic data with different tolerances for rotational
        # and translational coordinates
        output_mot = os.path.join(self.dataDir, 'Data', 'sync-tests', 
                                  'OpenSimData', 'Kinematics', 'walk.mot')
        ref_mot = os.path.join(self.dataDir, 'Data', 'sync-tests', 
                               'OutputReference', 'walk.mot')

        output_mot_df, output_mot_metadata = load_mot(output_mot)
        ref_mot_df, ref_mot_metadata = load_mot(ref_mot)

        pd.testing.assert_index_equal(output_mot_df.columns, ref_mot_df.columns)

        t0 = 1.0
        tf = 5.0
        compare_mot(self, output_mot_df, ref_mot_df, t0, tf)


if __name__ == "__main__":
    unittest.main()
