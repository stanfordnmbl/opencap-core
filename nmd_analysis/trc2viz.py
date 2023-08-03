
import sys
from pathlib import Path

import yaml
import numpy as np
import pandas as pd

from vispy import app, scene
from vispy.scene import visuals

import imageio

from utils import read_trc


# datapath = Path('/Volumes/GoogleDrive-112026393729621442608/My Drive/NMBL Lab/OpenCap for NMD biomarkers/data')
gdrive = Path('/Users/psr/Library/CloudStorage/GoogleDrive-paru@stanford.edu')
datadir = gdrive / 'My Drive/NMBL Lab/OpenCap for NMD biomarkers/data'
# datadir /= 'dataset03_dm_fshd'
datadir /= 'dataset02_fshd'

df_session = pd.read_excel(datadir / 'session_info.xlsx', )
df_trial = pd.read_excel(datadir / 'trial_info.xlsx')
df_info = df_session.merge(df_trial, on='sid')

pid = 'noID_002'
# sid = 'a1647b35-9f1e-4146-90dc-fb43eaaac659'
trial = '5xSTS'
# sid, trial = '''
# d83ce1e2-238d-4eaf-9ed7-04a876ddd3e5	arm_rom
# '''.strip().split()

df_temp = df_info[(df_info.pid == pid) & (df_info.trial == trial)]
# df_temp = df_info[(df_info.sid == sid) & (df_info.trial == trial)]
assert len(df_temp) == 1

sid = df_temp.sid.iloc[0]
fpath = datadir / f'opencap_data/OpenCapData_{sid}/MarkerData/{trial}.trc'
assert fpath.exists()


################################################################################


# length of gif to export in seconds
giflen = 0

# name of gif export file
saveto = Path("./trc2viz-demo.gif")

# integer factor to downsample framerate
ds_factor = 3


################################################################################

fps, markers, xyz = read_trc(fpath)

# reorder chans for vizualization
xyz[:,:,[0,1,2]] = xyz[:,:,[2,0,1]]
xyz[:,:,[0,1]] *= -1
N, M, _ = xyz.shape

# create canvas and view
canvas = scene.SceneCanvas(keys='interactive', size=(300,300),
                           # bgcolor=(1,1,1,1), show=True)
                           bgcolor=(1,1,1,1), show=True)
view = canvas.central_widget.add_view()

# create initial position (ensure data inside frame for whole preview)
xyz_cent = np.mean(xyz, axis=(0,1))
xyz_cent = np.broadcast_to(xyz_cent, xyz.shape)
locs = np.argmax(np.abs(xyz - xyz_cent), axis=0)
mlocs, dlocs = np.indices(locs.shape)
xyz_init = xyz[locs, mlocs, dlocs]

# plot initial vertices
pcolor = (0,0,0,1)
psize = 5
scatter = visuals.Markers()
scatter.set_data(xyz_init, edge_color=None, face_color=pcolor, size=psize)
view.add(scatter)

# # plot edges
# lcolor = (0,0,1,0.25)
# lwidth = 3
# lines = []
# sensor_graph = []
# with open(landgraphpath, 'r') as f:
#     for line in f.readlines():
#         if line.strip().startswith('#') or line.isspace():
#             continue
#         line_start, line_stop = tuple(line.strip().split())
#         p1 = landmark_names.index(line_start)
#         p2 = landmark_names.index(line_stop)
#         sensor_graph.append((p1, p2))
# for l, (p1, p2) in enumerate(sensor_graph):
#     line = visuals.Line(pos=positions[0,(p1,p2),:],
#             color=lcolor, width=lwidth, parent=canvas.scene)
#     lines.append(line)
#     view.add(line)

# view.camera = 'panzoom' # for 2D
view.camera = 'arcball'


if giflen > 0:
    imwriter = imageio.get_writer(saveto, mode='I', fps=20)

frame = 0
def update(event=None):
    global frame

    if frame < N and frame % ds_factor == 0:
        scatter.set_data(xyz[frame,:,:], edge_color=None, face_color=pcolor, size=psize)
        # for l, (p1, p2) in enumerate(sensor_graph):
        #     lines[l].set_data(pos=positions[frame,(p1,p2),:])

    # write frame to GIF
    if giflen > 0:
        if frame < fps * giflen and frame % 3 == 0:
            imwriter.append_data(canvas.render(alpha=True), )

    frame += 1

update()

timer = app.Timer(1/fps, connect=update, iterations=N, start=True)


if __name__ == '__main__': # and sys.flags.interactive == 0:
    app.run()
    app.quit()
    
    if giflen > 0:
        imwriter.close()



