import os, glob

import pyvista as pv
import numpy as np
from tqdm.auto import tqdm
from tqdm.contrib import tenumerate
from PIL import Image

theme = pv.themes.DarkTheme()
# theme.nan_color = pv.Color(color='w',opacity=0)
# pv.global_theme.font.color = 'black'
pv.set_plot_theme(theme)

# pv.global_theme.nan_color = pv.Color(color='w',opacity=0)

folder = r'X:\20220113_surface_sampling_samples\Hydrostone_BD_results'
# folder = r'X:\20220113_surface_sampling_samples\PLT'
dr_type = 'min_bidirectional'
dr_name = 'Minimum Bidirectional Delta T'
output_name = 'min_bidirectional.png'
# file_prefix = 'BD_91_1a.stl_deltastar_t_r2.5'
files = glob.glob(folder+"/*.stl")

pl = pv.Plotter(off_screen=True,line_smoothing=True)

scaling = int((2*3543)//pl.window_size[0])
pv.global_theme.font.size *= scaling
# pv.global_theme.font.title_size *= scaling
# pv.global_theme.font.label_size *= scaling
pl.window_size = (pl.window_size[0]*scaling,pl.window_size[1]*scaling)
x_translate = 52
y_translate = 28
order = 3
min_x = 1000
min_y = 1000
for i,file in tenumerate(files):
    tqdm.write(file)
    file_base = os.path.basename(file)
    file_prefix = file_base+"_delta_t_r2.5"
    direction_file = os.path.join(folder,file_prefix + "_directions.vtu")
    magnitude_file = os.path.join(folder,file_prefix + "_magnitude.vtu")
    
    # pl.add_axes()
    dir_data = pv.get_reader(direction_file).read()
    # mag_data = pv.get_reader(magnitude_file).read()
    ind = np.zeros([dir_data.n_points],dtype=bool)
    ind[::500] = True
    dir_glyphs = dir_data.extract_points(ind).glyph(scale=1,orient=dr_type,geom=pv.Line())
    dir_glyphs = dir_glyphs.translate(-np.array(dir_glyphs.center))
    dir_glyphs = dir_glyphs.translate([x_translate * (i % order),-(i//order)*y_translate,0])
    min_y = min(min_y,dir_glyphs.bounds[2])
    min_x = min(dir_glyphs.bounds[0],min_x)
    # mag_data.rename_array(dr_type,dr_name,'point')

    # pl.add_mesh(mag_data.threshold(),scalars=dr_name)
    pl.add_mesh(dir_glyphs,line_width=scaling/3.5,color='white')
pl.view_xy()
pl.enable_parallel_projection()
pl.add_ruler([min_x + 3,min_y-5,0],[min_x+13,min_y-5,0],title="10 mm",font_size_factor=0.3,label_format="%g",number_labels=3)

img = pl.screenshot(os.path.join(folder,output_name),transparent_background = True)
# img = 255 - img
# Image.fromarray(img).save(os.path.join(folder,output_name))
