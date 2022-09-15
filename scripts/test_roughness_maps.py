from surface_roughness import Surface, SampleWindow, roughness_map
import matplotlib.pyplot as plt

w = SampleWindow(is_circle=True,radius=2.5)
file_path = 'example_surface.stl'
def generate_map(surface,method,submethods,w:SampleWindow):
    map = roughness_map(surface,method,w,1,1)
    map.sample(verbose=True)
    map.evaluate()
    for submethod in submethods:
        map.analyze_directional_roughness(submethod)

        plt.figure(figsize=(6.5,4))
        map.plot_quiver(submethod,'min_bidirectional',ax=plt.subplot(221))
        map.plot_magnitude(submethod,'max_bidirectional',ax=plt.subplot(222))
        map.plot_distribution(submethod,'max_bidirectional',50,ax=plt.subplot(212))
        plt.tight_layout()
        plt.show()
    return map
        
surface = Surface(path=file_path)
map = generate_map(surface,'delta_t',['delta_t'],w)
map.to_vtk("test","delta_t")
pass