
# finds all shoulder trials to reprocess
# writes CV of (session ID, trial name) pairs

from pathlib import Path
import pandas as pd

datadir = Path('./datadir')
df_trial = pd.read_excel(datadir / 'trial_info.xlsx')
# df_trial = pd.read_excel(snakemake.input[0])

shoulder_trials = ['brooke', 'arm_rom']
df_shoulder = df_trial[df_trial.trial_clean.isin(shoulder_trials)]

df_shoulder[['sid', 'trial']].to_csv('shoulder_trials.csv',
                                     header=False, index=False)
# df_shoulder[['sid', 'trial']].to_csv(snakemake.output[0],
#                                      header=False, index=False)




