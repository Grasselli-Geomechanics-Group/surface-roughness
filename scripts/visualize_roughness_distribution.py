#%%
import numpy as np
import matplotlib.pyplot as plt
import mplstereonet as mpl
from surface_roughness.roughness import Surface

file = r'scripts\example_surface.stl'
mesh = Surface(file)
fig = plt.figure()
ax = fig.add_subplot(111,projection='stereonet')
x,y,z = mesh.normals.T

plunge,bearing = mpl.vector2plunge_bearing(x,y,z)
ax.density_contourf(plunge,bearing,measurement='lines')
ax.line(plunge,bearing,marker='o',color='black')
plt.show()

# %%
