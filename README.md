# surface-roughness

Surface roughness is a library that processes 3D surface STL data and produces oriented roughness metrics.

## Citation
If you wish to use this work, please cite this paper according to CITATION.cff.

## Installation

Installation is completed by 
```
pip install surface-roughness
```

## Usage
### Roughness mapping
```python
from surface_roughness import Surface, SampleWindow, roughness_map

# Load Surface STL File into Python
surface = Surface(path="example_surface.stl")

# Specify SampleWindow parameters
w = SampleWindow(is_circle=True,radius=2) # in units of STL mesh

# Generate roughness map object based on method presented in Magsipoc & Grasselli (2023) with parameters
map = roughness_map(
    surface,        # surface object required to interact with library
    'delta_t',      # Method for analyzing roughness
    w,              # SampleWindow object
    0.5,            # Distance between windows in mesh units
    1               # Number of vertices for mesh facet to be included in window sampling
    )

# Start subsampling process
map.sample()

# Evaluate roughness of subsamples
map.evaluate()

# Analyze directional roughness statistics
map.analyze_directional_roughness('delta_t')
map.analyze_directional_roughness('delta*_t')

# Pickle map to save analysis (optional)
with ('example_surface_r2_s0.5.pickle','wb') as f:
    pickle.dump(map, f)

# Save data to VTK for visualization
map.to_vtk('example_surface_r2_s0.5','delta_t')

```

### Directional Roughness
```python
from surface_roughness import Surface
import matplotlib.pyplot as plt     # For plotting graphs

surface = Surface('example_surface.stl')

# Calculate roughness from Tatone and Grasselli (2009) doi: 10.1063/1.3266964
surface.evaluate_thetamax_cp1()
az = surface.thetamax_cp1('az') # Get azimuth correlating with analysis
thetamaxcp1_roughness = surface.thetamax_cp1('thetamax_cp1')

# Plot surface roughness
plt.figure()
plt.polar(az,thetamaxcp1_roughness,label=r'$\theta^*_{max}/(C+1)$')
plt.legend()

# Calculate roughness from Babanouri and Karami Nasab (2017) doi: 10.1007/s00603-016-1139-1
surface.evaluate_delta_t()
az_t = s.delta_t('az')
delta_t = s.delta_t('delta_t')

az_a = s.delta_a('az')
delta_a = s.delta_a('delta_a')

az_n = s.delta_n('az')
delta_n = s.delta_n('delta_n')


plt.figure()
plt.polar(az_t,delta_t,label='$\Delta_T$')
plt.polar(az_a,delta_a,label='$\Delta_A$')
plt.polar(az_n,delta_n,label='$\Delta_N$')
plt.legend()

```

## License
[MIT](https://choosealicense.com/licenses/mit)