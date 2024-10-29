import requests
import time
import json
import os
import shutil
from utilsServer import processTrial, runTestSession
import traceback
import logging
import glob
from datetime import datetime, timedelta
import numpy as np
from utilsAPI import getAPIURL, getWorkerType, getASInstance, unprotect_current_instance, get_number_of_pending_trials
from utilsAuth import getToken
from utils import (getDataDirectory, checkTime, checkResourceUsage,
                  sendStatusEmail, checkForTrialsWithStatus,
                  getCommitHash, getHostname, postLocalClientInfo,
                  postProcessedDuration)

logging.basicConfig(level=logging.INFO)

API_TOKEN = getToken()
API_URL = getAPIURL()
workerType = getWorkerType()
autoScalingInstance = getASInstance()
logging.info(f"AUTOSCALING TEST INSTANCE: {autoScalingInstance}")

# if true, will delete entire data directory when finished with a trial
isDocker = True

# get start time
initialStatusCheck = False
t = time.localtime()

# For removing AWS machine scale-in protection
t_lastTrial = time.localtime()
justProcessed = True
with_on_prem = True
minutesBeforeRemoveScaleInProtection = 2
max_on_prem_pending_trials = 5

while True:
    # Run test trial at a given frequency to check status of machine. Stop machine if fails.
    if checkTime(t,minutesElapsed=30) or not initialStatusCheck:
        runTestSession(isDocker=isDocker)           
        t = time.localtime()
        initialStatusCheck = True

    # When using autoscaling, if there are on-prem workers, then we will remove
    # the instance scale-in protection if the number of pending trials is below
    # a threshold so that the on-prem workers are prioritized.
    if with_on_prem:
        # Query the number of pending trials        
        if autoScalingInstance:
            pending_trials = get_number_of_pending_trials()
            logging.info(f"Number of pending trials: {pending_trials}")
            if pending_trials < max_on_prem_pending_trials:
                # Remove scale-in protection and sleep in the cycle so that the
                # asg will remove that instance from the group.
                logging.info("Removing scale-in protection (out loop).")
                unprotect_current_instance()
                logging.info("Removed scale-in protection (out loop).")
                for i in range(3600):
                    time.sleep(1)
           
    # workerType = 'calibration' -> just processes calibration and neutral
    # workerType = 'all' -> processes all types of trials
    # no query string -> defaults to 'all'
    queue_path = "trials/dequeue/?workerType=" + workerType
    try:
        r = requests.get("{}{}".format(API_URL, queue_path),
                         headers = {"Authorization": "Token {}".format(API_TOKEN)})
    except Exception as e:
        traceback.print_exc()
        time.sleep(15)
        continue

    if r.status_code == 404:
        logging.info("...pulling " + workerType + " trials from " + API_URL)
        time.sleep(1)
        
        # When using autoscaling, we will remove the instance scale-in protection if it hasn't
        # pulled a trial recently and there are no actively recording trials
        if (autoScalingInstance and not justProcessed and 
            checkTime(t_lastTrial, minutesElapsed=minutesBeforeRemoveScaleInProtection)):
            if checkForTrialsWithStatus('recording', hours=2/60) == 0:
                # Remove scale-in protection and sleep in the cycle so that the
                # asg will remove that instance from the group.
                logging.info("Removing scale-in protection (in loop).")
                unprotect_current_instance()
                logging.info("Removed scale-in protection (in loop).")
                for i in range(3600):
                    time.sleep(1)
            else:
                t_lastTrial = time.localtime()
                
        # If a trial was just processed, reset the timer.
        if autoScalingInstance and justProcessed:
            justProcessed = False
            t_lastTrial = time.localtime()
            
        continue
    
    if np.floor(r.status_code/100) == 5: # 5xx codes are server faults
        logging.info("API unresponsive. Status code = {:.0f}.".format(r.status_code))
        time.sleep(5)
        continue
    
    # Check resource usage
    resourceUsage = checkResourceUsage(stop_machine_and_email=True)
    logging.info(json.dumps(resourceUsage))
    logging.info(r.text)
    
    trial = r.json()
    trial_url = "{}{}{}/".format(API_URL, "trials/", trial["id"])
    logging.info(trial_url)
    logging.info(trial)
    
    if len(trial["videos"]) == 0:
        error_msg = {}
        error_msg['error_msg'] = 'No videos uploaded. Ensure phones are connected and you have stable internet connection.'
        error_msg['error_msg_dev'] = 'No videos uploaded.'

        r = requests.patch(trial_url, data={"status": "error", "meta": json.dumps(error_msg)},
                         headers = {"Authorization": "Token {}".format(API_TOKEN)})
        continue

    # The following is now done in main, to allow reprocessing trials with missing videos
    # if any([v["video"] is None for v in trial["videos"]]):
    #     r = requests.patch(trial_url, data={"status": "error"},
    #                 headers = {"Authorization": "Token {}".format(API_TOKEN)})
    #     continue

    trial_type = "dynamic"
    if trial["name"] == "calibration":
        trial_type = "calibration"
    if trial["name"] == "neutral":
        trial["name"] = "static"
        trial_type = "static"

    logging.info("processTrial({},{},trial_type={})".format(trial["session"], trial["id"], trial_type))

    try:
        # Post new client info to Trial and start timer for processing duration
        postLocalClientInfo(trial_url)
        process_start_time = datetime.now()

        # trigger reset of timer for last processed trial              
        processTrial(trial["session"], trial["id"], trial_type=trial_type, isDocker=isDocker)

        # note a result needs to be posted for the API to know we finished, but we are posting them 
        # automatically thru procesTrial now
        r = requests.patch(trial_url, data={"status": "done"},
                         headers = {"Authorization": "Token {}".format(API_TOKEN)})
        logging.info('0.5s pause if need to restart.')
        time.sleep(0.5)

    except Exception as e:
        r = requests.patch(trial_url, data={"status": "error"},
                         headers = {"Authorization": "Token {}".format(API_TOKEN)})
        traceback.print_exc()

        # Antoine: Removing this, it is too often causing the machines to stop. Not because
        # the machines are failing, but because for instance the video is very long with a lot
        # of people in it. We should not stop the machine for that. Originally the check was
        # to catch a bug where the machine would hang, I have not seen this bug in a long time.
        # args_as_strings = [str(arg) for arg in e.args]
        # if len(args_as_strings) > 1 and 'pose detection timed out' in args_as_strings[1].lower():
        #     logging.info("Worker failed. Stopping machine.")
        #     message = "A backend OpenCap machine timed out during pose detection. It has been stopped."
        #     sendStatusEmail(message=message)
        #     raise Exception('Worker failed. Stopped.')
    
    finally:
        # End process duration timer and post duration to database
        process_end_time = datetime.now()
        postProcessedDuration(trial_url, process_end_time - process_start_time)

    justProcessed = True
    
    # Clean data directory
    if isDocker:
        folders = glob.glob(os.path.join(getDataDirectory(isDocker=True),'Data','*'))
        for f in folders:         
            shutil.rmtree(f)
            logging.info('deleting ' + f)