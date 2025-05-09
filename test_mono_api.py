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
# API_URL = getAPIURL()
API_URL = 'http://127.0.0.1:8000/'

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
    # if checkTime(t,minutesElapsed=30) or not initialStatusCheck:
    #     runTestSession(isDocker=isDocker)           
    #     t = time.localtime()
    #     initialStatusCheck = True

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
    queue_path = "trials/dequeue/"
    try:
        r = requests.get("{}{}".format(API_URL, queue_path),
                         headers = {"Authorization": "Token {}".format(API_TOKEN)})
        time.sleep(1)
    except:
        test = 1
        