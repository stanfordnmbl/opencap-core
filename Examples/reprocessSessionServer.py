# -*- coding: utf-8 -*-
"""
This script simulates collecting data by changing trial status and deleting results.

@author: suhlr
"""

import sys
import os
import requests
import time
import numpy as np
sys.path.append(os.path.abspath('./..'))

import utils
from decouple import config
from multiprocessing.dummy import Pool
import utilsAuth
import utilsAPI

# Authenticate
API_TOKEN = utilsAuth.getToken()
API_URL = utilsAPI.getAPIURL()

#%% List of inputs

# %% Functions
def getTrialStatus(trial_id):
    r = requests.get(API_URL+"trials/{}/".format(trial_id),
           headers = {"Authorization": "Token {}".format(API_TOKEN)}).json()
    return r['status']

def reprocessSession(session_id,waitToFinish=False,justDynamic=True,justErrors=False):
    print('reprocessing session ' + session_id)
    # get trials
    session = utils.getSessionJson(session_id)   
    statusData = {'status':'stopped'}
    
    if not justDynamic:
        # Calibrate and wait
        calibration_id = utils.getCalibrationTrialID(session_id)
        r = requests.patch(API_URL+"trials/{}/".format(calibration_id), data=statusData,
                      headers = {"Authorization": "Token {}".format(API_TOKEN)})
        utils.deleteResult(calibration_id)
        
        print('session: ' + session_id + ' reprocessing calibration')
        while getTrialStatus(calibration_id) in ['stopped','processing']:
            time.sleep(1)
            
        # Neutral and wait
        neutral_id = utils.getNeutralTrialID(session_id)
        r = requests.patch(API_URL+"trials/{}/".format(neutral_id), data=statusData,
                     headers = {"Authorization": "Token {}".format(API_TOKEN)})
        utils.deleteResult(neutral_id)
        
        print('session: ' + session_id + ' reprocessing neutral')
        while getTrialStatus(neutral_id) in ['stopped','processing']:
            time.sleep(1)   
        
        status = [getTrialStatus(calibration_id), getTrialStatus(neutral_id)]
        
    trialNameList = []
    for c_t, trial in enumerate(session['trials']):
        # Only reprocess trials with errors
        if justErrors and not session['trials'][c_t]['status'] == 'error':
            continue    

        # if 'test-240_1' not in trial['name']:
        #     continue

        # if 'test_11' not in trial['name']:
        #     continue


        

        if trial['name'] not in ['calibration','neutral']:
            print('session: ' + session_id + ' reprocessing ' + trial['name'])
            trialNameList.append(trial['id'])
            utils.deleteResult(trial['id'])
            # change status to stopped
            r = requests.patch(API_URL+'trials/{}/'.format(trial['id']), data=statusData,
                         headers = {"Authorization": "Token {}".format(API_TOKEN)})
    
    if waitToFinish:
        while any([getTrialStatus(tN) in ['stopped','processing'] for tN in trialNameList]):
            time.sleep(5)
        
        [status.append(getTrialStatus(tID)) for tID in trialNameList]

    else:
        status = []
    
    return status

                   

def simulateCollection(nSessions=1,waitToFinish=False,justDynamic=True):   
    # Random sessions to reprocess
    session_ids = ['62fc7cae-a05b-4d19-8dba-73d01f40f4e8',
                   '6fc75175-7a66-4fe0-819d-c5f2ee8672dc',
                   '5fe119d7-cdb4-4803-ba35-78150a98f58a',
                   '5e94d4f8-e266-4dc7-a66a-bc34cd9fbd8b']# This fourth one has 28 trials...
    if 'dev' in API_URL: # trial for dev pipeline
        session_ids = ['9d19e9c1-0f06-463f-bb6a-b20cd33493c5']
        
    nSessions = np.min([nSessions,len(session_ids)])
    
    pool = Pool(nSessions) 

    futures = []
    for session_id in session_ids[:nSessions]:
        futures.append(pool.apply_async(reprocessSession, [session_id],
                                        {'waitToFinish':waitToFinish,'justDynamic':justDynamic}))
    statuses = []
    if waitToFinish: 
        for future in futures:
            statuses = statuses + future.get()
        print("{} of {} trials succeeded.".format(
               np.sum([s == 'done' for s in statuses]),len(statuses)))
    pool.close()  
            
    return statuses   
    
                
def actLikePhones(nSeconds=20,waitForResponses=True,nThreads=10,delay=False):
    if 'dev' in API_URL:
        raise Exception('Using the dev API. Switch env vars.')
    randomDeviceID = '7d1828c3-a195-4699-88b4-c7ad8a8d2ae0'
    randomSessionID = '62fc7cae-a05b-4d19-8dba-73d01f40f4e8'
    phoneURL = API_URL + 'sessions/{}/status/?device_id={}'.format(randomSessionID,randomDeviceID)
    
    print('Pinging API like phones for {}sec.'.format(nSeconds))
    pool=Pool(nThreads) # multithread for faster request generation
    tic = time.time()
    n = 0
    n500 = 0
    futures = []
    
    while time.time()-tic < nSeconds:
        for i in range(nThreads):
            futures.append(pool.apply_async(requests.get, [phoneURL]))  
            if delay:
                time.sleep(.5/nThreads) # simulates phones perfectly separated in time
        if waitForResponses:
            for future in futures:
                if future.get().status_code != 200:
                    print(future.get().status_code)
                    n500 +=1
                    print('request group ' + str(n))
        n += nThreads    
    pool.close()

    print("{:.1f} pings/s. {} total, {} failed.".format(n/nSeconds,n,n500))
        
def downloadSessions(nSessions=5,waitForResponse=False):
    if 'dev' in API_URL:
        raise Exception('Using the dev API. Switch env vars.')
    sessionList = ['6eca46e3-6cad-4a13-8956-a9fcd64ebc1b',
                   '6a3d1c0d-6356-4a54-b2c1-c0d4f71ab5c3',
                   '16fd9007-8c2c-41bf-94d9-d5723a64c2e3',
                   '95a8ac40-5dd0-4a47-ad02-9eb2e8e75b24',
                   '015d6935-c86f-46b9-ace5-d4c1c99d3882']
    nSessions = np.min([nSessions,len(sessionList)]) 
    print('Simulating download of {} sessions'.format(nSessions))
        
    pool = Pool(nSessions) 

    futures = []
    for session_id in sessionList[:nSessions]:
        futures.append(pool.apply_async(requests.get, [API_URL + 'sessions/{}/download'.format(session_id)],
                                        {'headers':{"Authorization": "Token {}".format(API_TOKEN)}}))
    if waitForResponse: 
        for future in futures:
            print(future.get())
    pool.close()
                
#%% Script
# Recreate data collection. 
#    waitToFinish = True if you want to see outcome of reprocessing
#    justDynamic = False if you want to process calib and neutral. Can load up backend
#                  faster if this is False, b/c all dynamic trials start w/o waiting for
#                  calib/neutral to finish.
#    if nSessions = 4, the 4th session has 28 trials, so will really stack the queue
# simulateCollection(nSessions=1,waitToFinish=True,justDynamic=False)

# sessions = ['f1975763-d2c6-4e42-b725-cce50b4215dd', '4c59e643-8b07-4c5d-b0fc-48b56b04130e', 'b584a184-4d8e-417d-b0cd-56102ab1e25c']


# 30s at 240Hz
sessions = ['bb06d157-4007-427c-b8a7-49daf0f47688']

for session in sessions:
    reprocessSession(session,waitToFinish=False,justDynamic=False,justErrors=False)

test=1

# # as long as waitForResponse = False, can run this prior to act like phones to test simultaneously
# downloadSessions(nSessions=1,waitForResponse=False) 

# actLikePhones(nSeconds=10,nThreads = 16, delay=True)


        