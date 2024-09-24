import sys
import os
import requests
import time
sys.path.append(os.path.abspath('./..'))
import utils
import utilsAuth
import utilsAPI

# %% Authenticate
API_TOKEN = utilsAuth.getToken()
API_URL = utilsAPI.getAPIURL()

# %% Functions
def getTrialStatus(trial_id):
    r = requests.get(API_URL+"trials/{}/".format(trial_id),
           headers = {"Authorization": "Token {}".format(API_TOKEN)}).json()
    return r['status']


def reprocessData(
        session_id, waitToFinish=False, justNeutral=False, justDynamic=False, 
        justDynamicErrors=False, reprocessCalibration=False, onlySpecifiedTrials=None, 
        status_to_set='reprocess'):

    print('reprocessing session ' + session_id)
    session = utils.getSessionJson(session_id)   
    statusData = {'status': status_to_set}

    if reprocessCalibration:
        # Calibration and wait
        calibration_id = utils.getCalibrationTrialID(session_id)
        _ = requests.patch(API_URL+"trials/{}/".format(calibration_id), data=statusData,
                            headers = {"Authorization": "Token {}".format(API_TOKEN)})
        utils.deleteResult(calibration_id)            
        print('session: ' + session_id + ' reprocessing calibration')
        while getTrialStatus(calibration_id) in [status_to_set, 'processing']:
            time.sleep(5)
        print('calibration trial reprocessing finished')
    
    if not justDynamic:
        # Neutral and wait
        neutral_id = utils.getNeutralTrialID(session_id)
        _ = requests.patch(API_URL+"trials/{}/".format(neutral_id), data=statusData,
                           headers = {"Authorization": "Token {}".format(API_TOKEN)})
        utils.deleteResult(neutral_id)        
        print('session: ' + session_id + ' reprocessing neutral')
        while getTrialStatus(neutral_id) in [status_to_set, 'processing']:
            time.sleep(5)
        print('neutral trial reprocessing finished')

    if not justNeutral:
        trialNameList = []
        for c_t, trial in enumerate(session['trials']):
            # Trials with errors only
            if justDynamicErrors and not session['trials'][c_t]['status'] == 'error':
                continue            
            # Dynamic trials only
            if trial['name'] not in ['calibration','neutral']:
                # Specific trials only
                if onlySpecifiedTrials and trial['name'] not in onlySpecifiedTrials:
                    continue
                print('session: ' + session_id + ' reprocessing ' + trial['name'])
                trialNameList.append(trial['id'])
                utils.deleteResult(trial['id'])
                _ = requests.patch(API_URL+'trials/{}/'.format(trial['id']), data=statusData, 
                                   headers = {"Authorization": "Token {}".format(API_TOKEN)})
    
    if waitToFinish:
        while any([getTrialStatus(tN) in [status_to_set, 'processing'] for tN in trialNameList]):
            time.sleep(5)
        print('dynamic trial(s) reprocessing finished')

if __name__ == '__main__':
    # %% Reprocess data
    sessions = [
        '04dee2de-3284-43ee-8abd-a792a1304d71'
    ]

    # Parameters:
    #   reprocessCalibration = True if you want to reprocess the calibration trial (this ignores justNeutral and justDynamic)
    #   justNeutral = True is you want to reprocess the neutral trial only (and the calibration trial if reprocessCalibration is True)
    #   justDynamic = True if you want to reprocess the dynamic trial(s) only (and the calibration trial if reprocessCalibration is True)
    #   justDynamicErrors = True if you want to reprocess the dynamic trial(s) with errors only (and the calibration trial if reprocessCalibration is True)
    #   onlySpecifiedTrials = [] if you want to reprocess all dynamic trial(s) (all dynamic trial(s) with errors), or a list of trial names to reprocess
    #   status_to_set = 'reprocess' or 'stopped'. Trials with status 'reprocess' will be processed after trials with status 'stopped'.
    #       Please set to 'reprocess' if you want to reprocess a lot of trials, otherwise this might hinder data processing of other OpenCap users.
    #   waitToFinish = True if you want the function to wait until all dynamic trials are reprocessed
    for session in sessions:
        reprocessData(
            session, reprocessCalibration=False, justNeutral=False, justDynamic=True,
            justDynamicErrors=False, onlySpecifiedTrials=[], status_to_set='stopped',
            waitToFinish=False)
            
