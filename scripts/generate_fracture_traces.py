import glob
import os
import pickle

from surface_roughness import Surface, SampleWindow, roughness, roughness_map
from surface_roughness._profile import power_law, powerlaw_fitting,Profile
import numpy as np
import matplotlib.pyplot as plt

w = SampleWindow(is_circle=True,radius=2.5)
# file = 'scripts/example_surface.stl'
# file = r'X:\20220113_surface_sampling_samples\20220823_beza_direct_tension_bot_1_1.stl'
# file = r'X:\20220113_surface_sampling_samples\MontneyCCNBD2018\Petronas_M3-3-No2_CCNBD_side1_trimmed.stl'
# file = r'X:\20220113_surface_sampling_samples\Hydrostone_BD_results\BD_72_1a.stl'
# file = r'X:\20220113_surface_sampling_samples\MontneyBD2018\BD_montney_MG3_2_2LG4_top_intact_1_1_cropped.stl'
# file = r'X:\20220113_surface_sampling_samples\MontneyBD2018\BD_montney_MG3_2_2LG4_bot_broken_1_1_cropped.stl'
file = r'X:\20220113_surface_sampling_samples\MontneyBD2018\BD_montney_MG3_2_1LG4_bot_intact_1_1_cropped.stl'
surface = Surface(path=file)
pickle_suffix = '_r2.5.pickle'
n_profiles = 20
if os.path.exists(f'{file}{pickle_suffix}'):
    with open(f'{file}{pickle_suffix}','rb') as f:
        map = pickle.load(f)
else:
    map = roughness_map(surface,'delta_t',w,1,1)
    map.sample(verbose=True)
    map.evaluate()
    map.analyze_directional_roughness('delta_t')
    with open(f'{file}{pickle_suffix}','wb') as f:
        pickle.dump(map,f)
# Obtain tracer profile
fig,samples = map.generate_spaced_streamlines('delta_t','min_bidirectional',n_profiles)
fig.savefig(f'{file}_r{map.sample_window.radius}_parallel_profile_spaced.svg')
profile_par = []
hhcorr_par = []
for i,sample in enumerate(samples):
    if len(sample) > 3:
        profile_par.append(surface.sample_profile(sample))
        np.savetxt(f'{file}_r{map.sample_window.radius}_par_profile_{i}.csv',profile_par[-1].points,delimiter=',')
        print(profile_par[-1].z2())
        # hhcorr_par.append(profile_par[-1].hhcorr())
        # np.savetxt(f'{file}par_hhcorr_{i}.csv',hhcorr_par[-1],delimiter=',')

fig,samples = map.generate_spaced_streamlines('delta_t','minperp_bidirectional',n_profiles)
fig.savefig(f'{file}_r{map.sample_window.radius}_perp_profile_spaced.svg')
profile_perp = []
hhcorr_perp = []
for i,sample in enumerate(samples):
    if len(sample) > 3:
        profile_perp.append(surface.sample_profile(sample))
        np.savetxt(f'{file}_r{map.sample_window.radius}_perp_profile_{i}.csv',profile_perp[-1].points,delimiter=',')
        print(profile_perp[-1].z2())
        # hhcorr_perp.append(profile_perp[-1].hhcorr())
        # np.savetxt(f'{file}perp_hhcorr_{i}.csv',hhcorr_perp[-1],delimiter=',')



