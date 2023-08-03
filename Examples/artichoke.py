
from pathlib import Path

import pandas as pd

import sys, os
sys.path.append(os.path.abspath('./..'))
from utils import download_session

session_id_fpath = Path('./session_ids_dhd_1.csv')
download_path = Path('./Data')

df_id = pd.read_csv(session_id_fpath, header=None, names=['sid'])

sids = df_id.sid.dropna()
for sid in sids:
    print(sid)
    download_session(sid, download_path)

