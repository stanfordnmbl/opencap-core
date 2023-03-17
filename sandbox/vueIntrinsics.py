# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 12:08:00 2023

test intrinsics from video

@author: hpl
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import utilsChecker as uc

CheckerBoardParams = {
    'dimensions': (11,8),
    'squareSize': 60}


video_dir = 'C:/Users/hpl/Documents/MyRepositories/opencap-core_ACL/sandbox/'
video_name = 'vueCheckerIntrinsics.2122194.mov'
uc.video2Images(os.path.join(video_dir,video_name),nImages=50)
CamParams = uc.calcIntrinsics(video_dir, CheckerBoardParams=CheckerBoardParams,
                           filenames=['*.jpg'], 
                           saveFileName=os.path.join(video_dir,'cameraIntrinsics.pickle'),
                           visualize = False)
test =1