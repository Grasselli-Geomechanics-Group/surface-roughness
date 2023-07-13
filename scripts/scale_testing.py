#%%
import glob
import os
import pickle

from surface_roughness import Surface, SampleWindow, roughness, roughness_map
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
sns.set_style('whitegrid')
sns.set_context("paper",font_scale=1)
sns.color_palette("colorblind")
file = r'X:/20220113_surface_sampling_samples/20220823_beza_direct_tension_top_1_1.stl'
# file = r'X:\20220113_surface_sampling_samples\Hydrostone_BD_results\BD_91_1a.stl'
# file = r'X:\20220113_surface_sampling_samples\Hydrostone_BD_results\BD_113_1a+2a.stl'

surface = Surface(path=file)
radii = [10,8,4,2.5,1.5,1.,0.5]
surface.evaluate_delta_t()
print(surface.delta_t('delta_t'))
#%%

for radius in radii:
    if os.path.exists(f'{file}pickledump_{radius}.pickle'):
        continue
    w = SampleWindow(is_circle=True,radius=radius)
    rmap = roughness_map(surface,'delta_t',w,radius/2,1)
    rmap.sample(verbose=True)
    rmap.evaluate()
    rmap.analyze_directional_roughness('delta_t')
    # map.to_vtk(f'{file}_delta_t_r{w.radius}','delta_t',True)
    # map.to_csv(f'{file}_delta_t_r{w.radius}.csv')

    with open(f'{file}pickledump_{radius}.pickle','wb') as f:
        pickle.dump(rmap,f)
    
#%%
df = pd.DataFrame()
for radius in radii:
    data = pd.read_csv(f'{file}_delta_t_r{radius}.csv')
    data['Sample Radius [mm]'] = radius
    df = pd.concat([df,data])
    print(radius, df['max_unidirectional_magnitude-delta_t'].max())
df = df.rename(columns={
    "max_unidirectional_magnitude-delta_t":"Maximum Unidirectional Delta T",
    "max_bidirectional_magnitude-delta_t":"Maximum Bidirectional Delta T",
    "minperp_unidirectional_magnitude-delta_t":"Minimum-Perpendicular Unidirectional Delta T",
    "minperp_bidirectional_magnitude-delta_t":"Minimum-Perpendicular Bidirectional Delta T"})

#%%
fig,ax = plt.subplots(figsize=(3.5433,3.5433*3/4))
ax.set_xlim([5,15])
ax.set_xticks(np.arange(5,17,2))
sns.kdeplot(df,x="Maximum Unidirectional Delta T",hue="Sample Radius [mm]",common_norm=False,palette=sns.color_palette("colorblind"))
# %%
fig,ax = plt.subplots(figsize=(3.5433,3.5433*3/4))

sns.boxenplot(df,x="Sample Radius [mm]",y="Maximum Unidirectional Delta T",ax=ax)
ax.set_ylim([5,15])
ax.set_yticks(np.arange(5,17,2))

# %%
