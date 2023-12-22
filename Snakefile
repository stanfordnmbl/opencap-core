
from pathlib import Path
import sys
sys.path.append(Path('./..'))
import utils

import pandas as pd

datadir = Path('./datadir')

# df_session = pd.read_excel(datadir / 'session_info.xlsx')
df_trial = pd.read_excel(datadir / 'trial_info.xlsx')
sids = df_trial[~df_trial.trial_clean.isna()].sid.unique()

OUTPUTS = [f'{datadir}/opencap_data/{sid}' for sid in sids]


rule all:
    input:
        OUTPUTS

# TODO this should be multithreaded with async instead of multiprocessed w/ snakemake cores

rule download_session:
    # input:
    #     trc='{datadir}/opencap_data/{sid}/MarkerData/PostAugmentation/{trial}/{trial}.trc',
    #     sto='{datadir}/opencap_data/{sid}/OpenSimData/Dynamics/{trial}_shoulder.sto'
    output:
        directory('{datadir}/opencap_data/{sid}')
    script:
        'download_session.py'

