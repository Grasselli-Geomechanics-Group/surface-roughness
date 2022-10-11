import glob
import os

from surface_roughness import Surface, SampleWindow, roughness, roughness_map
from surface_roughness._profile import power_law, powerlaw_fitting,Profile
import numpy as np
import matplotlib.pyplot as plt

w = SampleWindow(is_circle=True,radius=2.5)
# file = 'scripts/example_surface.stl'
file = r'X:\20220113_surface_sampling_samples\20220930_Beza_direct_tension_2_top_1_1.stl'
surface = Surface(path=file)
map = roughness_map(surface,'delta_t',w,1,1)
map.sample(verbose=True)
map.evaluate()
map.analyze_directional_roughness('delta_t')
# Obtain tracer profile
fig,samples = map.generate_spaced_streamlines('delta_t','min_bidirectional',6)
fig.savefig(f'{file}_parallel_profile_spaced.png')
profile_par = []
hhcorr_par = []
for i,sample in enumerate(samples):
    profile_par.append(surface.sample_profile(sample))
    np.savetxt(f'{file}par_profile_{i}.csv',profile_par[-1].points,delimiter=',')
    hhcorr_par.append(profile_par[-1].hhcorr())
    np.savetxt(f'{file}par_hhcorr_{i}.csv',hhcorr_par[-1],delimiter=',')

fig,samples = map.generate_spaced_streamlines('delta_t','minperp_bidirectional',6)
fig.savefig(f'{file}_perp_profile_spaced.png')
profile_perp = []
hhcorr_perp = []
for i,sample in enumerate(samples):
    profile_perp.append(surface.sample_profile(sample))
    np.savetxt(f'{file}perp_profile_{i}.csv',profile_perp[-1].points,delimiter=',')
    hhcorr_perp.append(profile_perp[-1].hhcorr())
    np.savetxt(f'{file}perp_hhcorr_{i}.csv',hhcorr_perp[-1],delimiter=',')



