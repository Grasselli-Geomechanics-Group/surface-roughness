import glob
import os
import pickle

from surface_roughness import Surface, SampleWindow, roughness, roughness_map
from surface_roughness._profile import power_law, powerlaw_fitting
import numpy as np
import matplotlib.pyplot as plt

w = SampleWindow(is_circle=True,radius=2.5)
# file = 'scripts/example_surface.stl'
# file = r'X:\20220113_surface_sampling_samples\20220930_Beza_direct_tension_2_top_1_1.stl'
file = r'X:\20220113_surface_sampling_samples\MontneyBD2018\BD_montney_MG3_2_2LG4_top_intact_1_1_cropped.stl'
surface = Surface(path=file)
pickle_suffix = '.pickle'
if os.path.exists(f'{file}{pickle_suffix}'):
    with open(f'{file}{pickle_suffix}','rb') as f:
        map = pickle.load(f)
else:
    map = roughness_map(surface,'delta_t',w,1,1)
    map.sample(verbose=True)
    map.evaluate()
    map.analyze_directional_roughness('delta_t')
fig,samples = map.generate_streamline_selection('delta_t','min_bidirectional')
fig.savefig(f'{file}_parallel_profile.png')
profile_par = surface.sample_profile(samples)
hhcorr_par = profile_par.hhcorr()
np.savetxt(f'{file}parallel_profile.csv',profile_par,delimeter=',')
np.savetxt(f'{file}parallel_hhcorr.csv',hhcorr_par,delimiter=',')
par_parm,par_std = powerlaw_fitting(hhcorr_par)
powerfig,ax = plt.subplots()
ax.loglog(hhcorr_par[:,0],hhcorr_par[:,1],'x')
ax.loglog(hhcorr_par[:,0],power_law(hhcorr_par[:,0],par_parm[0],par_parm[1]))

print(f'beta = {par_parm[1]}+/-{par_std[1]}')
fig,samples = map.generate_streamline_selection('delta_t','minperp_bidirectional')
fig.savefig(f'{file}_perp_profile.png')
profile_perp = surface.sample_profile(samples)
hhcorr_perp = profile_perp.hhcorr()
np.savetxt(f'{file}perp_profile.csv',profile_perp,delimiter=',')
np.savetxt(f'{file}perp_hhcorr.csv',hhcorr_perp,delimiter=',')
perp_parm,perp_std = powerlaw_fitting(hhcorr_perp)
ax.loglog(hhcorr_perp[:,0],hhcorr_perp[:,1],'o')
ax.loglog(hhcorr_perp[:,0],power_law(hhcorr_perp[:,0],perp_parm[0],perp_parm[1]))
ax.legend(['Par Measurement','Par Fit','Perp Measurement','Perp Fit'])
powerfig.savefig(f'{file}_profile_powerlaw.png')
print(f'beta = {perp_parm[1]}+/-{perp_std[1]}')

