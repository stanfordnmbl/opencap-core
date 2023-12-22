
# finds all shoulder trials to reprocess
# writes CV of (session ID, trial name) pairs

from pathlib import Path
import pandas as pd

datadir = Path('./datadir')
df_trial = pd.read_excel(datadir / 'trial_info.xlsx')

shoulder_trials = ['brooke', 'arm_rom']
df_shoulder = df_trial[df_trial.trial_clean.isin(shoulder_trials)]


def sto_path(row):
    return datadir / f'opencap_data/{sid}/OpenSimData' \
                      'Dynamics/{trial}_shoulder.sto'
df_shoulder['sto_fpath'] = df_shoulder.apply(sto_path, axis=1).to_list()

for 

