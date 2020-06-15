#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 17:27:38 2020

@author: weixijia
"""
import xml.etree.ElementTree as ET
import math
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import subprocess
from IPython.display import HTML, Image

data = pd.read_csv('sensor_wifi_timestep100_14.csv',encoding='gb2312')

x=data[['14']]
y=data[['15']]


#plot path
plt.scatter(x,y,s=1)
plt.show()
n=2000


X=x[0:n]
Y=y[0:n]
Z=rss_total[0:n]

#plot path
plt.plot(X,Y)
plt.show()
#plot rss distribution
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(Y, X, Z, cmap=plt.cm.viridis, linewidth=0.2)
surf=ax.plot_trisurf(Y, X, Z, cmap=plt.cm.viridis, linewidth=0.2)
fig.colorbar( surf, shrink=0.5, aspect=5)
ax.view_init(30, 45)
ax.plot_trisurf(Y, X, Z, cmap=plt.cm.jet, linewidth=0.01)
plt.show()
#plot gif animation of the rss distribution
for angle in range(70,210,2):
 
    # Make the plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(Y, X, Z, cmap=plt.cm.viridis, linewidth=0.2)
     
    # Set the angle of the camera
    ax.view_init(30,angle)
 
# Save it
filename='PNG/ANIMATION/Volcano_step'+str(angle)+'.png'
plt.savefig(filename, dpi=96)
plt.gca()

#plot git
# # Create a figure and a 3D Axes
# rc('animation', html='html5')
# fig = plt.figure()
# ax = Axes3D(fig)

# def init():
#     # Plot the surface.
#     ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
#     return fig,

# def animate(i):
#     # azimuth angle : 0 deg to 360 deg
#     ax.view_init(elev=10, azim=i*4)
#     return fig,

# # Animate
# ani = animation.FuncAnimation(fig, animate, init_func=init,
#                                frames=90, interval=50, blit=True)

# fn = 'rotate_azimuth_angle_3d_surf'
# ani.save(fn+'.mp4',writer='ffmpeg',fps=1000/50)
# ani.save(fn+'.gif',writer='imagemagick',fps=1000/50)


# ani.save('animation.gif', writer='imagemagick', fps=60)
# Image(url='animation.gif')

#plot path after m sample (show the last round)
m=2000

p=x[m:,]
q=y[m:,]
plt.plot(p,q)
plt.show()
