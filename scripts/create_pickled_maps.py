import os, glob, pickle
import numpy as np
import matplotlib.pyplot as plt

from surface_roughness.sampling import RoughnessMap

# files = glob.glob("X:/20220113_surface_sampling_samples/MontneyBD2018/*stl.pickle")
# files = glob.glob("X:/20220113_surface_sampling_samples/*.stl.pickle")
# files = glob.glob("X:/20220113_surface_sampling_samples/Hydrostone_BD_results/BD_31/*.stl.pickle")
# files = glob.glob(r"X:/20220113_surface_sampling_samples/MontneyCCNBD2018/Petronas_M3-3-No2_CCNBD_side1_trimmed*.pickle")
files = glob.glob(r"X:\20220113_surface_sampling_samples\PLT\*.pickle")


# files = files[-4:]

for file in files:
    output_prefix = os.path.splitext(file)[0]
    rmap:RoughnessMap = None
    with open(f'{file}','rb') as f:
        rmap = pickle.load(f)
    for submethod in ['delta_t']:
        w = rmap.sample_window
        
        # plt.figure(figsize=(6.5,4))
        # rmap.plot_quiver(submethod,'min_bidirectional',ax=plt.subplot(221))
        # rmap.plot_magnitude(submethod,'max_bidirectional',ax=plt.subplot(222))
        # rmap.plot_distribution(submethod,'max_bidirectional',50,ax=plt.subplot(212))
        # plt.tight_layout()
        submethod_savestr = submethod.replace('*','star')


        # plt.savefig(f'{output_prefix}_{submethod_savestr}_r{w.radius}_roughnessmap.svg')

        rmap.to_vtk(f'{output_prefix}_{submethod_savestr}_r{w.radius}',submethod,find_edges=True,method='nearest')
        # rmap.to_csv(submethod,f'{output_prefix}_{submethod_savestr}_r{w.radius}.csv')
        # rmap.print_directional_roughness(f'{output_prefix}_dr_{w.radius}','delta_t')