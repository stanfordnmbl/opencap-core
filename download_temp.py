


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


# if __name__ == '__main__':
#     # datadir = Path(snakemake.output[0])
#     download_session(snakemake.wildcards['sid'], snakemake.output[0])



sid = '3f3e9f24-aa8e-40c7-97dd-67325e70c2cd'

sdir = Path('Data') / sid
download_session(sid, sdir)
