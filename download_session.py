
from pathlib import Path

import utils

def download_session(sid, sdir):
    if Path(sdir).exists():
        return
    utils.downloadSomeStuff(sid,
                        justDownload=True,
                        isDocker=False,
                        data_dir=Path(sdir).parent,
                        includeVideos=False)


if __name__ == '__main__':
    # datadir = Path(snakemake.output[0])
    download_session(snakemake.wildcards['sid'], snakemake.output[0])



# sid = '9ce91ceb-2bf8-4544-ae22-3da5ad3cb9a1'

# sdir = Path('datadir') / 'opencap_data' / sid
# download_session(sid, sdir)
