#%%
import glob
import os
import pickle

from surface_roughness import Surface, SampleWindow, roughness, roughness_map
import numpy as np
import matplotlib.pyplot as plt
radius=2
w = SampleWindow(is_circle=True,radius=radius)

files = glob.glob("C:/Users/emags/OneDrive - University of Toronto/20230717_ccnbd_roughness/*.stl")

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

    # generate_map(surface,file,'delta_t',['delta_t','delta*_t'],w)
    generate_map(surface,file,'delta_t',['delta_t'],w)
    
    # generate_map(surface,'thetamax_cp1',['thetamax_cp1'],w)
    # generate_map(surface,'delta_a',['delta_a','delta*_a'],w)
    # generate_map(surface,'mean_dip',['mean_dip','std_dip'],w)

# %%
