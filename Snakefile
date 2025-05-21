
from pathlib import Path
import sys
sys.path.append(Path('./..'))
import utils

import pandas as pd

datadir = Path('./datadir')

# df_session = pd.read_excel(datadir / 'session_info.xlsx')
df_trial = pd.read_excel(datadir / 'trial_info.xlsx')

print(len(df_trial))
# df_trial = df_trial.iloc[:2611]
# df_trial = df_trial.iloc[:1958]
# df_trial = df_trial.iloc[:1632]
# df_trial = df_trial.iloc[:1620]
# df_trial = df_trial.iloc[:1591]
# df_trial = df_trial.iloc[:1570]
# df_trial = df_trial.iloc[:1560]
# df_trial = df_trial.iloc[:1555]
# df_trial = df_trial.iloc[:1554]
# df_trial = df_trial.iloc[:1553]
# df_trial = df_trial.iloc[:1552]
# df_trial = df_trial.iloc[:1550]
# df_trial = df_trial.iloc[:1469]
# df_trial = df_trial.iloc[:1305]
# print(df_trial.iloc[1552])


sids = df_trial[~df_trial.trial_clean.isna()].sid.unique()

OUTPUTS = [f'{datadir}/opencap_data/{sid}' for sid in sids]

rule all:
    input:
        OUTPUTS
    script:
        'rename_models.py'

# TODO this should be multithreaded with async instead of multiprocessed w/ snakemake cores

rule download_session:
    output:
        directory('{datadir}/opencap_data/{sid}')
    script:
        'download_session.py'


