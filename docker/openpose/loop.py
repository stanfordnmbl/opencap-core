import os
import time
import logging
import shutil

logging.basicConfig(level=logging.INFO)

logging.info("Waiting for data...")

video_path = "/openpose/data/video_openpose.mov"
output_dir = "/openpose/data/output_openpose"

# Set resolution for OpenPose ('default', '1x736', or '1x1008_4scales').
resolutionPoseDetection = '1x736'
# Adjust OpenPose call based on selected resolution.    
if resolutionPoseDetection == 'default':
    cmd_hr = ' '
elif resolutionPoseDetection == '1x1008_4scales':
    cmd_hr = ' --net_resolution "-1x1008" --scale_number 4 --scale_gap 0.25 '
elif resolutionPoseDetection == '1x736':
    cmd_hr = ' --net_resolution "-1x736" '
elif resolutionPoseDetection == '1x736_2scales':
    cmd_hr = ' --net_resolution "-1x736" --scale_number 2 --scale_gap 0.75 '
    
if os.path.isfile(video_path):
    os.remove(video_path)

while True:    
    if not os.path.isfile(video_path):
        time.sleep(0.1)
        continue

    logging.info("Processing...")

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    command = "/openpose/build/examples/openpose/openpose.bin\
        --video {video_path}\
        --display 0\
        --write_json {output_dir}\
        --render_pose 0{cmd_hr}".format(video_path=video_path, output_dir=output_dir, cmd_hr=cmd_hr)
    os.system(command)

    logging.info("Done. Cleaning up")
    os.remove(video_path)
