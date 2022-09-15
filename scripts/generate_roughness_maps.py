import glob
import os

from surface_roughness import Surface, SampleWindow, roughness, roughness_map
import numpy as np
import matplotlib.pyplot as plt

w = SampleWindow(is_circle=True,radius=2.5)
# files = ['example_surface.stl']
# files = glob.glob("X:/20220113_surface_sampling_samples/Hydrostone_BD_results/BD_31/*.stl")
# files = glob.glob("X:/20220113_surface_sampling_samples/*.stl")
# files = glob.glob("X:/20220113_surface_sampling_samples/Hydrostone_BD_results/*.stl")
# files = glob.glob("X:/20220113_surface_sampling_samples/MontneyCCNBD2018/*.stl")
# files.extend(glob.glob("X:/20220113_surface_sampling_samples/Hydrostone_BD_results/BD_31/*.stl"))
# files.extend(glob.glob("X:/20220113_surface_sampling_samples/PLT/*.stl"))
# files.extend(glob.glob("X:/20220113_surface_sampling_samples/20220718_EM_alberta_roadcut_sample_1_1_s*.stl"))
# files = glob.glob("X:/20220113_surface_sampling_samples/PLT/*.stl")
files = glob.glob("X:/20220113_surface_sampling_samples/roadcut_results/*.stl")
 
def generate_map(surface,method,submethods,w:
    SampleWindow):
    map = roughness_map(surface,method,w,1,1)
    map.sample(verbose=True)
    map.evaluate()
    for submethod in submethods:
        map.analyze_directional_roughness(submethod)

        # plt.figure(figsize=(6.5,4))
        # map.plot_quiver(submethod,'min_bidirectional',ax=plt.subplot(221))
        # map.plot_magnitude(submethod,'max_bidirectional',ax=plt.subplot(222))
        # map.plot_distribution(submethod,'max_bidirectional',50,ax=plt.subplot(212))
        # plt.tight_layout()
        submethod_savestr = submethod.replace('*','star')

        # plt.savefig(f'{file}_{submethod_savestr}_r{w.radius}_roughnessmap.svg')
        map.to_vtk(f'{file}_{submethod_savestr}_r{w.radius}',submethod)
        map.to_csv(f'{file}_{submethod_savestr}_r{w.radius}.csv')
        map.print_directional_roughness(file,submethod)

for file in files:
    # if os.path.exists(f'{file}_delta_t_r2.5_directions.vtu'):
        # continue
    surface = Surface(path=file)

    generate_map(surface,'delta_t',['delta_t','delta*_t'],w)
    
    # generate_map(surface,'thetamax_cp1',['thetamax_cp1'],w)
    # generate_map(surface,'delta_a',['delta_a','delta*_a'],w)
    # generate_map(surface,'mean_dip',['mean_dip','std_dip'],w)
