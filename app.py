import requests
import time
import json
from utilsServer import processTrial
import traceback
import logging
import numpy as np
from utilsAPI import getAPIURL
from utilsAuth import getToken

logging.basicConfig(level=logging.INFO)

API_TOKEN = getToken()
API_URL = getAPIURL()

def trc2json(trc_path, output_path):
    with open(trc_path, "r") as file_motion:
        lines = file_motion.readlines()

    start = 0
    while lines[start][:6] != "Frame#":
        start += 1

    markers = lines[start].split("\t")[2:]
    columns = lines[start+1].split("\t")[2:]

    times = []
    observations = []
    
    for line in lines[start+2:]:
        line_elements = line.rstrip().split("\t")
        if len(line_elements) < 5:
            continue
        times.append(float(line_elements[1]))
        observations.append(list(map(float, line_elements[2:])))

    res = {
        "framerate": 60,
        "time": times,
        "markers": markers,
        "colnames": columns,        
        "data": observations
    }
 
    with open(output_path, "w") as file_output:
        file_output.write(json.dumps(res))
        
    return output_path

while True:
    queue_path = "trials/dequeue/"
    try:
        r = requests.get("{}{}".format(API_URL, queue_path),
                         headers = {"Authorization": "Token {}".format(API_TOKEN)})
    except Exception as e:
        traceback.print_exc()
        time.sleep(15)
        continue

    if r.status_code == 404:
        logging.info(".")
        time.sleep(0.5)
        continue
    
    if np.floor(r.status_code/100) == 5: # 5xx codes are server faults
        logging.info("API unresponsive. Status code = {:.0f}.".format(r.status_code))
        time.sleep(5)
        continue

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

    if any([v["video"] is None for v in trial["videos"]]):
        r = requests.patch(trial_url, data={"status": "error"},
                     headers = {"Authorization": "Token {}".format(API_TOKEN)})
        break

    trial_type = "dynamic"
    if trial["name"] == "calibration":
        trial_type = "calibration"
    if trial["name"] == "neutral":
        trial["name"] = "static"
        trial_type = "static"

    logging.info("processTrial({},{},trial_type={})".format(trial["session"], trial["id"], trial_type))

    try:
        processTrial(trial["session"], trial["id"], trial_type=trial_type, isDocker=True)   
        # note a result needs to be posted for the API to know we finished, but we are posting them 
        # automatically thru procesTrial now
        r = requests.patch(trial_url, data={"status": "done"},
                         headers = {"Authorization": "Token {}".format(API_TOKEN)})
        
        print('0.5s pause if need to restart.')
        time.sleep(0.5)
    except:
        r = requests.patch(trial_url, data={"status": "error"},
                         headers = {"Authorization": "Token {}".format(API_TOKEN)})
        traceback.print_exc()
