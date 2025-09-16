
from pathlib import Path

import utils

def download_session(sid, sdir):
    if Path(sdir).exists():
        return
    # utils.downloadSomeStuff(sid,
    #                     justDownload=True,
    #                     isDocker=False,
    #                     data_dir=Path(sdir).parent,
    #                     includeVideos=False)
    utils.download_session(
            sid,
            sessionBasePath=Path(sdir).parent,
            zipFolder=False,
            writeToDB=False,
            downloadVideos=False,
    )


# if __name__ == '__main__':
#     download_session(snakemake.wildcards['sid'], snakemake.output[0])


sid = '5e6be6c9-f5d0-4005-90e2-86064e1ec629'
sdir = Path('datadir') / 'opencap_data' / sid
download_session(sid, sdir)
