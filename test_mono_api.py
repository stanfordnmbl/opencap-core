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


API_TOKEN = getToken()
# API_URL = getAPIURL()
API_URL = 'http://127.0.0.1:8000/'

isMono = "True"



           
# workerType = 'calibration' -> just processes calibration and neutral
# workerType = 'all' -> processes all types of trials
# no query string -> defaults to 'all'
queue_path = "trials/dequeue/?isMono=" + isMono
# queue_path = "trials/dequeue/"

r = requests.get("{}{}".format(API_URL, queue_path),
                     headers = {"Authorization": "Token {}".format(API_TOKEN)})

# get num calib cameras
session_id = '2c862324-6a4e-401c-9b62-382c1853903b'
r_n_cameras = requests.get("{}/sessions/{}/get_n_calibrated_cameras/".format(API_URL, session_id),
                     headers = {"Authorization": "Token {}".format(API_TOKEN)})