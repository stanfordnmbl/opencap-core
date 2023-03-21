# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 20:19:39 2023

@author: suhlr
"""

# make animation of keypoints in main
# TODO DELETE THIS
import matplotlib.pyplot as plt
import matplotlib.animation as animation
def makeAnimation(keypoints2d):
    #animation of keypoints2d
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 1920), ylim=(0, 1080))
    line, = ax.plot([], [],'.', ls='')
    def init():
        line.set_data([], [])
        return line,
    def animate(i):
        x = keypoints2d[:,i,0]
        y = keypoints2d[:,i,1]
        line.set_data(x, y)
        return line,
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                    frames=keypoints2d.shape[1], interval=20, blit=True)
    
    return anim

anim = makeAnimation(keypoints2D['Cam0'])
#anim.save('test.mp4', fps=frameRate, extra_args=['-vcodec', 'libx264'])
plt.show()