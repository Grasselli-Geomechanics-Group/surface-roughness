#%%
import glob
import os
import pickle

from surface_roughness import Surface, SampleWindow, roughness, roughness_map
import numpy as np
import matplotlib.pyplot as plt
radius=2.5
w = SampleWindow(is_circle=True,radius=radius)
# files = [r'scripts\example_surface.stl']
# files = [r'X:\20220113_surface_sampling_samples\Hydrostone_BD_results\BD_113_1a+2a.stl']
# files = glob.glob("X:/20220113_surface_sampling_samples/Hydrostone_BD_results/BD_31/*.stl")
# files = [r'X:/20220113_surface_sampling_samples/20220823_beza_direct_tension_top_1_1.stl']
# files = glob.glob("X:/20220113_surface_sampling_samples/*.stl")
# files = glob.glob("X:/20220113_surface_sampling_samples/MontneyBD2018/*.stl")
# files = glob.glob("X:/20220113_surface_sampling_samples/MontneyBD2018/BD_montney_MG3_2_2LG4_top_intact_1_1_cropped.stl")
#files = glob.glob('C:/Users/emags/OneDrive - University of Toronto/Roughness_maps/*.stl')
# files = glob.glob("X:/20220113_surface_sampling_samples/Hydrostone_BD_results/*.stl")
# files = glob.glob("X:/20220113_surface_sampling_samples/MontneyCCNBD2018/Petronas_M3-3-No2_CCNBD_side1_trimmed.stl")
# files.extend(glob.glob("X:/20220113_surface_sampling_samples/MontneyCCNBD2018/*.stl"))
# files.extend(glob.glob("X:/20220113_surface_sampling_samples/Hydrostone_BD_results/*.stl"))
# files.extend(glob.glob("X:/20220113_surface_sampling_samples/Hydrostone_BD_results/BD_31/*.stl"))
# files.extend(glob.glob("X:/20220113_surface_sampling_samples/PLT/*.stl"))
# files.extend(glob.glob("X:/20220113_surface_sampling_samples/roadcut_results/*.stl"))
# files = glob.glob("X:/20220113_surface_sampling_samples/PLT/*.stl")
# files = glob.glob("X:/20220113_surface_sampling_samples/roadcut_results/*.stl")
files = glob.glob("C:/Users/emags/OneDrive - University of Toronto/20230700_megablock/*.stl")
 
def generate_map(surface,file,method,submethods,w:SampleWindow):
    print(f"Analyzing {file}")
    if os.path.exists(f'{file}_r{radius}.pickle'):
        with open(f'{file}_r{radius}.pickle','rb') as f:
            map = pickle.load(f)
    else:
        map = roughness_map(surface,method,w,radius/2.5,1)
    # map.to_vtk(f'{file}__r{w.radius}','delta_t',find_edges=True)
        map.sample(verbose=True)
        map.evaluate()
    for submethod in submethods:
        if not os.path.exists(f'{file}_r{radius}.pickle'):
            map.analyze_directional_roughness(submethod)

        # plt.figure(figsize=(6.5,4))
        # map.plot_quiver(submethod,'min_bidirectional',ax=plt.subplot(221))
        # map.plot_magnitude(submethod,'max_bidirectional',ax=plt.subplot(222))
        # map.plot_distribution(submethod,'max_bidirectional',50,ax=plt.subplot(212))
        # plt.tight_layout()
        submethod_savestr = submethod.replace('*','star')


        # plt.savefig(f'{file}_{submethod_savestr}_r{w.radius}_roughnessmap.svg')
        if not os.path.exists(f'{file}_{submethod_savestr}_r{w.radius}_roughness.vtu'):
            map.to_vtk(f'{file}_{submethod_savestr}_r{w.radius}',submethod,find_edges=True)
        # map.to_csv(f'{file}_{submethod_savestr}_r{w.radius}.csv')
        # map.print_directional_roughness(file,submethod)
    if not os.path.exists(f'{file}_r{radius}.pickle'):
        with open(f'{file}_r{radius}.pickle','wb') as f:
            pickle.dump(map,f)
    
for file in files:
    # if os.path.exists(f'{file}_delta_t_r2.5_directions.vtu'):
        # continue
    surface = Surface(path=file)

    generate_map(surface,file,'delta_t',['delta_t','delta*_t'],w)
    
    # generate_map(surface,'thetamax_cp1',['thetamax_cp1'],w)
    # generate_map(surface,'delta_a',['delta_a','delta*_a'],w)
    # generate_map(surface,'mean_dip',['mean_dip','std_dip'],w)

# %%
