
import os
from pathlib import Path

datadir = Path('./datadir')

if __name__ == '__main__':
    for fpath in datadir.glob('opencap_data/*/OpenSimData/Model/LaiArnoldModified2017_poly_withArms_weldHand_scaled.osim'):
        new_fpath = fpath.parent / 'LaiUhlrich2022_scaled.osim'
        os.rename(fpath, new_fpath)


