from itertools import compress
import os
from typing import Union
from collections import deque

import numpy as np
from pandas import DataFrame, concat
from tqdm import tqdm
from tqdm.contrib import tenumerate
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backend_bases import MouseButton

import meshio

from scipy.stats import gaussian_kde
from scipy.interpolate import griddata

from surface_roughness.roughness_impl import (
    _rs,
    _cppDirectionalRoughness,
    _DirectionalRoughness_Evaluator,
    _PyDirectionalRoughness,
    _PyTINBasedRoughness,
    _cppTINBasedRoughness,
    _TINBasedRoughness_Evaluator,
    _cppTINBasedRoughness_againstshear,
    _TINBasedRoughness_againstshear_Evaluator,
    _cppTINBasedRoughness_bestfit,
    _TINBasedRoughness_bestfit_Evaluator,
    _cppMeanDipRoughness,
    _MeanDipRoughness_Evaluator,
    DirRoughnessBase
)
from surface_roughness.roughness import Surface
from surface_roughness._sampling_utils import (
    _resample_polyline,
    _next_point,
    _comb_vectorfield,
    _generate_vtkdir_data,
    _centroids_in_offset_bounds,
    _triangles_from_p_in_sample,
    _in_circle,
    _offset_bounds,
    _streamline
)


class SampleWindow:
    """A class describing a sampling window to calculate local roughness of a 3D surface
    """
    def __init__(self,is_circle=True,radius=None,width=None,height=None):
        """Creates a SampleWindow based on provided arguments

        :param is_circle: Sets window as circle, defaults to True
        :type is_circle: bool, optional
        :param radius: Sets window radius if is_circle is True, defaults to None
        :type radius: float, optional
        :param width: Sets window width if is_circle is False, defaults to None
        :type width: float, optional
        :param height: Sets window height if is_circle is False, defaults to None
        :type height: float, optional
        """
        if radius is None and (width is None or height is None):
            raise ValueError("Either size arguments or width + height arguments must be defined")
        if is_circle and ((width is not None or height is not None) or radius is None):
            raise ValueError("Circle window must only have radius specified")
        
        self.is_circle = is_circle
        if is_circle:
            self.radius = radius
        else:
            self.width = width
            self.height = height

def roughness_map(surface:Surface,
    roughness_method:str,
    sample_window:SampleWindow,
    sample_spacing:float,
    sample_vertex_inclusion:int,
    seed_left_offset=0,
    seed_bot_offset=0,**roughness_kwargs):
    """Generates a RoughnessMap instance to handle sampling and roughness analysis

    :param surface: Surface data object to be processed
    :type surface: Surface
    :param roughness_method: Method to analyze by roughness map
    :type roughness_method: str
    :param sample_window: Sample window description to analyze local roughness
    :type sample_window: SampleWindow
    :param sample_spacing: Spacing between sample windows
    :type sample_spacing: float
    :param sample_vertex_inclusion: Minimum number of vertices to consider triangle in SampleWindow
    :type sample_vertex_inclusion: int
    :param seed_left_offset: Offset sample grid to the left, defaults to 0
    :type seed_left_offset: int, optional
    :param seed_bot_offset: Offset sample grid to the top, defaults to 0
    :type seed_bot_offset: int, optional
    :return: Returns a RoughnessMap object to handle sampling
    :rtype: RoughnessMap
    """
    if sample_vertex_inclusion > 3 or sample_vertex_inclusion < 0:
        raise ValueError("Argument sample_vertex_inclusion must be in range between 0-3")
    
    return RoughnessMap(surface,roughness_method,sample_window,sample_spacing,sample_vertex_inclusion,seed_left_offset,seed_bot_offset,**roughness_kwargs)


class RoughnessMap:
    """A class providing utilities to map local roughness of a 3D surface
    """
    def __init__(self,*args,**kwargs):
        self.surface:Surface = args[0]
        self.roughness_method:str = args[1]
        self.sample_window:SampleWindow = args[2]
        self.sample_spacing = args[3]
        self.sample_vertex_inclusion = args[4]
        self.seed_left_offset = args[5]
        self.seed_bot_offset = args[6]
        self.roughness_kwargs = kwargs
        self._methods = {
            'mean_dip':_cppMeanDipRoughness,
            'delta_t':_cppTINBasedRoughness,
            'delta_n':_cppTINBasedRoughness_againstshear,
            'delta_a':_cppTINBasedRoughness_bestfit,
            'thetamax_cp1':_cppDirectionalRoughness
        }
        self._evaluators = {
            'mean_dip':_MeanDipRoughness_Evaluator,
            'delta_t':_TINBasedRoughness_Evaluator,
            'delta_n':_TINBasedRoughness_againstshear_Evaluator,
            'delta_a':_TINBasedRoughness_bestfit_Evaluator,
            'thetamax_cp1':_DirectionalRoughness_Evaluator
        }
        self._submethods = {
            'mean_dip':['mean_dip'],
            'delta_t':['delta_t','delta*_t'],
            'delta_n':['delta_n','delta*_n'],
            'delta_a':['delta_a','delta*_a'],
            'thetamax_cp1':['thetamax_cp1']
        }
        if not self.roughness_method in self._methods:
            raise ValueError(f"Roughness method for roughness map must be {self._methods.keys()}")
        
        self.roughness_data = {}
        self.roughness_data_x2 = {}
        self.min_roughness = {}
        self.min_roughness_dir = {}
        self.max_roughness = {}
        self.max_roughness_dir = {}
        self.minperp_roughness = {}
        self.minperp_roughness_dir = {}
        self.min_roughness_x2 = {}
        self.min_roughness_dir_x2 = {}
        self.max_roughness_x2 = {}
        self.max_roughness_dir_x2 = {}
        self.minperp_roughness_x2 = {}
        self.minperp_roughness_dir_x2 = {}

        self.diropts = {
            'min_unidirectional':self.min_roughness_dir,
            'max_unidirectional':self.max_roughness_dir,
            'minperp_unidirectional':self.minperp_roughness_dir,
            'min_bidirectional':self.min_roughness_dir_x2,
            'max_bidirectional':self.max_roughness_dir_x2,
            'minperp_bidirectional':self.minperp_roughness_dir_x2
        }
        self.magopts = {
            'min_unidirectional':self.min_roughness,
            'max_unidirectional':self.max_roughness,
            'minperp_unidirectional':self.minperp_roughness,
            'min_bidirectional':self.min_roughness_x2,
            'max_bidirectional':self.max_roughness_x2,
            'minperp_bidirectional':self.minperp_roughness_x2
        }
        self.label = {
            'min_bidirectional':'Min. 2DR',
            'max_bidirectional':'Max. 2DR',
            'minperp_bidirectional':'Min. Perp. 2DR',
            'min_unidirectional':'Min. DR',
            'max_unidirectional':'Max. DR',
            'minperp_unidirectional':'Min. Perp. DR'
        }
    
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_evaluators']
        del state['_methods']
        return state

    def __setstate__(self,state):
        self.__dict__.update(state)
        self._methods = {
            'mean_dip':_cppMeanDipRoughness,
            'delta_t':_cppTINBasedRoughness,
            'delta_n':_cppTINBasedRoughness_againstshear,
            'delta_a':_cppTINBasedRoughness_bestfit,
            'thetamax_cp1':_cppDirectionalRoughness
        }
        self._evaluators = {
            'mean_dip':_MeanDipRoughness_Evaluator,
            'delta_t':_TINBasedRoughness_Evaluator,
            'delta_n':_TINBasedRoughness_againstshear_Evaluator,
            'delta_a':_TINBasedRoughness_bestfit_Evaluator,
            'thetamax_cp1':_DirectionalRoughness_Evaluator
        }

    def _reorient_points(self,samples,verbose):
        iter = tenumerate(list(samples)) if verbose else samples
        p_in_sample = [None]*len(samples)
        for i,sample in iter:
            if np.any(sample):
                p = self.surface.points[sample]
                
                normal = Surface._calculate_surface_normal(p)
                rot_mat = Surface._find_rotmatrix_2_z_pos(normal)
                centroid = p.mean(axis=0)
                p = (p-centroid) @ rot_mat.T + centroid
                point_indices = np.where(sample)[0][_in_circle(p[:,0],p[:,1],self.samples[i,0],self.samples[i,1],self.sample_window.radius)]
                p_in_sample[i] = np.zeros(sample.size,dtype=np.bool_)
                p_in_sample[i][point_indices] = True
            else:
                p_in_sample[i] = np.zeros(sample.size,dtype=np.bool_)
        
        return np.array(p_in_sample)   

    def sample(self,oriented=True,verbose=False):
        """Generate samples based on the sample window and spacing provided

        :param metric:
        :param oriented: Samples points with respect to normal of localized area, defaults to True
        :type oriented: bool, optional
        :param verbose: Enables command line output, defaults to False
        :type verbose: bool, optional
        """
        if verbose:
            print("Sampling...")
        bounds = self.surface.bounds()
        n = (bounds[1,:] - bounds[0,:]) / self.sample_spacing
        n = np.ceil(n).astype(np.int)+1
        xpoints = np.linspace(
            bounds[0,0]+self.seed_left_offset,
            bounds[0,0]+self.seed_left_offset+self.sample_spacing*n[0],
            n[0],endpoint=False)
        ypoints = np.linspace(
            bounds[0,1]+self.seed_bot_offset,
            bounds[0,1]+self.seed_bot_offset+self.sample_spacing*n[1],
            n[1],endpoint=False)
        x,y = np.meshgrid(xpoints,ypoints)
        self.samples = np.vstack([x.ravel(),y.ravel()]).T

        if verbose:
            print("Finding samples...")
        expansion_factor = 1 if not oriented else 1.2
        if self.sample_window.is_circle:
            if verbose:
                print("Point search")
                sample_iter = tqdm(self.samples,total=self.samples.shape[0])
            else:
                sample_iter = self.samples
            p_in_sample = np.array([_in_circle(
                self.surface.points[:,0],
                self.surface.points[:,1],
                sample[0],
                sample[1],self.sample_window.radius*expansion_factor) for sample in sample_iter])
            
            if oriented:
                if verbose:
                    print("Reorienting points")
                p_in_sample = self._reorient_points(p_in_sample,verbose)

            if verbose:
                print("Triangle search")
                p_in_sample_iter = tqdm(p_in_sample,total=len(p_in_sample))
            else:
                p_in_sample_iter = p_in_sample
            self.t_in_circle = np.array([_triangles_from_p_in_sample(
                self.surface.triangles,
                p
            ) for p in p_in_sample_iter],dtype=np.bool_)

            
            self.t_in_circle = [np.nonzero(t_in_c)[0] for t_in_c in self.t_in_circle]
            filter_list = [len(t) > 200 for t in self.t_in_circle]

            self.t_in_circle = list(compress(self.t_in_circle,filter_list))
            self.samples = self.samples[filter_list,:]
    
    def evaluate(self,folder:Union[str,bytes,os.PathLike]=None,file_prefix:str=None):
        """Enable the calculation of roughness parameters after sampling. 

        :param folder: Folder to output sampled surfaces, defaults to None
        :type folder: str, path, optional
        :param file_prefix: File prefix for output sampled surfaces, defaults to None
        :type file_prefix: str, optional
        """
        if folder:
            if not os.path.exists(folder):
                os.mkdir(folder)
        # Load data and run
        def run_calc(i,tlist):
            calc = self._methods[self.roughness_method](self.surface.points,self.surface.triangles,tlist)
            if folder is not None:
                file_name = os.path.join(folder,f"{file_prefix}_{i}.stl")
                calc.evaluate(False,file_name)
            else:
                calc.evaluate(False)
            return calc
        print("Analyzing sampled roughness...")
        evaluator = self._evaluators[self.roughness_method](self.surface.points,self.surface.triangles)
        calculators = evaluator.evaluate(self.t_in_circle)
        raw_roughness_calculators = [DirRoughnessBase(impl=impl) for impl in calculators]
        # self.raw_roughness_calculators:list[DirRoughnessBase] = [run_calc(i,tlist) for i,tlist in tqdm(enumerate(self.t_in_circle),total=len(self.t_in_circle))]

        # collect sample properties
        self.final_orientations = np.array([c.final_orientation for c in raw_roughness_calculators])
        self.min_bounds = np.array([c.min_bounds for c in raw_roughness_calculators])
        self.max_bounds = np.array([c.max_bounds for c in raw_roughness_calculators])
        self.centroids = np.array([c.centroid for c in raw_roughness_calculators])
        self.shape_sizes = np.array([c.shape_size for c in raw_roughness_calculators])
        self.total_areas = np.array([c.total_area for c in raw_roughness_calculators])
        
        self.az = raw_roughness_calculators[0]['az'][:,0]
        self.n_tri = [len(t_in_c) for t_in_c in self.t_in_circle]
        self.n_tri_dir = np.vstack([c['n_tri'].T for c in raw_roughness_calculators])

        for metric in self._submethods[self.roughness_method]:
            self.roughness_data[metric] = np.vstack([c[metric].T for c in tqdm(raw_roughness_calculators)])
        
    def analyze_directional_roughness(self,metric:str):
        """Process the roughness results for plotting

        :param metric: Metric to analyze directional roughness
        :type metric: str
        """

        print("Collecting stats")
        # Collect bidirectional data
        gt = self.az >= np.pi-10**-12
        lt = self.az < np.pi-10**-12
        self.n_tri_dir_x2 = self.n_tri_dir[:,gt] +  self.n_tri_dir[:,lt]
        self.roughness_data_x2[metric] = self.roughness_data[metric][:,gt] +  self.roughness_data[metric][:,lt]
        
        # Unidirectional roughness stats
        self.min_roughness[metric] = np.amin(self.roughness_data[metric],axis=1)
        self.min_roughness_dir[metric] = self.az[np.argmin(self.roughness_data[metric],axis=1)]

        self.minperp_roughness_dir[metric] = self.min_roughness_dir[metric] + np.pi/2
        minperp_idx = np.argmin(np.abs(self.roughness_data[metric]-self.minperp_roughness_dir[metric][:,np.newaxis]),axis=1)
        self.minperp_roughness[metric] = np.array([self.roughness_data[metric][row,idx] for row,idx in enumerate(minperp_idx)])
        
        self.max_roughness[metric] = np.amax(self.roughness_data[metric],axis=1)
        self.max_roughness_dir[metric] = self.az[np.argmax(self.roughness_data[metric],axis=1)]
        

        # Bidirectional roughness stats
        self.min_roughness_x2[metric] = np.amin(self.roughness_data_x2[metric],axis=1)
        self.min_roughness_dir_x2[metric] = self.az[np.argmin(self.roughness_data_x2[metric],axis=1)]

        self.minperp_roughness_dir_x2[metric] = self.min_roughness_dir_x2[metric] + np.pi/2
        minperp_idx = np.argmin(np.abs(self.roughness_data_x2[metric]-self.minperp_roughness_dir_x2[metric][:,np.newaxis]),axis=1)
        self.minperp_roughness_x2[metric] = np.array([self.roughness_data_x2[metric][row,idx] for row,idx in enumerate(minperp_idx)])
        
        self.max_roughness_x2[metric] = np.amax(self.roughness_data_x2[metric],axis=1)
        self.max_roughness_dir_x2[metric] = self.az[np.argmax(self.roughness_data_x2[metric],axis=1)]

    def plot_sample(self,sample_num,ax=None,**fig_kwargs):
        """Plots the sample in a Matplotlib figure

        :param sample_num: Index of sample to be plotted
        :type sample_num: int
        :return: matplotlib.pyplot.subplots figure handle
        :rtype: Figure
        :return: matplotlib.pyplot.subplots axes handle
        :rtype: axes.Axes
        """
        tri = self.surface.triangles[self.t_in_circle[sample_num]]
        if ax is None:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
        ax.plot_trisurf(
            self.surface.points[:,0],
            self.surface.points[:,1],
            self.surface.points[:,2], triangles=tri)

    def plot_quiver(self,metric:str,stat:str,quiver_loc='centroid',ax=None,**fig_kwargs):
        """Create quiver plot of selected data

        :param data: Directional data to plot {'min_unidirectional','max_unidirectional','min_bidirectional','max_bidirectional'}
        :type data: str

        :return: matplotlib.pyplot.subplots figure handle
        :rtype: Figure
        :return: matplotlib.pyplot.subplots axes handle
        :rtype: axes.Axes
        """
        quiver_loc_positions = {
            'centroid':self.centroids,
            'sample':self.samples
        }
        if ax is None:
            fig,ax = plt.subplots(**fig_kwargs)
        ax.quiver(
            quiver_loc_positions[quiver_loc][:,0],quiver_loc_positions[quiver_loc][:,1],
            np.cos(self.diropts[stat][metric]),np.sin(self.diropts[stat][metric]),
            headaxislength=1,headwidth=1,headlength=1)
        ax.set_title(f'{self.label[stat]} Direction')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.axis('equal')

    def plot_polar_roughness(self,metric:str,sample_num:int,ax=None,**fig_kwargs):
        """Plot directional roughness of a specific sample

        :param metric: Roughness metric used for plotting
        :type metric: str
        :param sample_num: Sample index to be used
        :type sample_num: int
        :return: matplotlib.pyplot.subplots figure handle
        :rtype: Figure
        :return: matplotlib.pyplot.subplots axes handle
        :rtype: axes.Axes
        """
        if ax is None:
            fig, ax = plt.subplots(subplot_kw={'projection':'polar'},**fig_kwargs)
        ax.plot(self.az,self.roughness_data[metric][sample_num])
        return fig,ax

    def plot_distribution(self,metric:str,stat:str,n_bins,ax=None,**fig_kwargs):
        """Plot KDE and histogram distribution of data

        :param metric: Roughness metric used for plotting
        :type metric: str
        :param stat: Statistic option from RoughnessMap.diropts.keys()
        :type stat: str
        :param n_bins: Number of bins in histogram
        :type n_bins: int
        :return: matplotlib.pyplot.subplots return parameters
        :rtype: Figure, axes.Axes
        """
        magopts = self.magopts[stat][metric]
        ind = ~np.isnan(magopts)
        kern = gaussian_kde(magopts[ind])
        x_range = np.linspace(np.amin(magopts[ind]),np.amax(magopts[ind]))
        pdf = kern.evaluate(x_range)
        if ax is None:
            fig,ax = plt.subplots(**fig_kwargs)
        ax.hist(self.magopts[stat][metric],bins=n_bins,density=True)
        ax.plot(x_range,pdf)
        ax.set_title(f'{self.label[stat]} magnitude distribution')
        ax.set_xlabel(f'{self.label[stat]} values')
        ax.set_ylabel('Density')

    def plot_circular_distribution(self,metric:str,stat:str,n_bins,ax=None,**fig_kwargs):
        """Plot rose diagram showing binned directions of requested roughness data

        :param metric: Roughness metric used for plotting
        :type metric: str
        :param stat: Statistic option from RoughnessMap.diropts.keys()
        :type stat: str
        :param n_bins: Number of bins to generate the histogram
        :type n_bins: int
        :return: matplotlib.pyplot.subplots figure handle
        :rtype: Figure
        :return: matplotlib.pyplot.subplots axes handle
        :rtype: axes.Axes
        """
        if ax is None:
            fig,ax = plt.subplots(subplot_kw={'projection':'polar'},**fig_kwargs)
        if stat in ['min_bidirectional','max_bidirectional']:
            max_range = np.pi
        else:
            max_range = 2*np.pi
        radii, theta = np.histogram(self.diropts[stat][metric],bins=n_bins,range=(0,max_range))
        theta = np.diff(theta)/2 + theta[:-1]
        width = np.pi/n_bins
        ax.bar(theta,radii,width=width,bottom=0.0)

    def plot_magnitude(self,metric:str,stat:str,n_colours=50,colorbar_label=None,ax=None,**fig_kwargs):
        """Plots the magnitude of requested roughness data.

        Args:
            metric (str): Roughness metric used for plotting
            stat (str):Statistic option from RoughnessMap.diropts.keys()
            n_colours (int, optional): Number of colours to use for the colorbar label. Defaults to 50.
            colorbar_label (colorbar_label, optional): Colorbar label. Defaults to None.
            ax (axes.Axes, optional): Plot on an existing axis instead of creating a new figure. Defaults to None.
        
        Returns:
            Matplotlib figure and axes
        """
        if ax is None:
            fig,ax = plt.subplots(**fig_kwargs)
        magopts = self.magopts[stat][metric]
        ind = ~np.isnan(magopts)
        
        mag = ax.tricontourf(self.samples[ind,0],self.samples[ind,1],magopts[ind],levels=n_colours,extend='neither')
        cb = plt.colorbar(mag)
        cb.set_label(colorbar_label)

        ax.set_title(f'{self.label[stat]} magnitude')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.axis('equal')
    
    def to_csv(self,metric:str,*args,**kwargs):
        """Creates a CSV of map data
        *args, **kwargs for pandas dataframe outputs including filename and to_csv options
        """
        def _build_df_row(idx):
            # flatten dataframe to row
            data = {str(az):datum for az,datum in zip(self.az,self.roughness_data[metric])}
            sample_df:DataFrame = DataFrame(data)
            sample_df = sample_df.unstack().to_frame().sort_index(level=1).T
            sample_df.columns = [x+'_'+str(round(y*180/np.pi)) for x,y in sample_df.columns]
            
            # add properties
            for dir,val in zip(['X','Y','Z'],self.final_orientations[idx]):
                sample_df[f'FinalOrientation_{dir}'] = val

            for dir,val in zip(['left','bot'],self.min_bounds[idx]):
                sample_df[f'Bounds_{dir}'] = val
            for dir,val in zip(['right','top'],self.max_bounds[idx]):
                sample_df[f'Bounds_{dir}'] = val
            for dir,val in zip(['X','Y','Z'],self.centroids[idx]):
                sample_df[f'Centroid_{dir}'] = val
            for dir,val in zip(['X','Y','Z'],self.shape_sizes[idx]):
                sample_df[f'Size_{dir}'] = val

            sample_df['Area'] = self.total_areas[idx]

            return sample_df

        dfs = concat([_build_df_row(c) for c in range(self.final_orientations.shape[0])])
        for key in self.max_roughness.keys():
            dfs['max_unidirectional_magnitude-'+key] = self.max_roughness[key]
            dfs['max_unidirectional_orientation-'+key] = self.max_roughness_dir[key]
            dfs['max_bidirectional_magnitude-'+key] = self.max_roughness_x2[key]
            dfs['max_bidirectional_orientation-'+key] = self.max_roughness_dir_x2[key]

            dfs['min_unidirectional_magnitude-'+key] = self.min_roughness[key]
            dfs['min_unidirectional_orientation-'+key] = self.min_roughness_dir[key]
            dfs['min_bidirectional_magnitude-'+key] = self.min_roughness_x2[key]
            dfs['min_bidirectional_orientation-'+key] = self.min_roughness_dir_x2[key]

            dfs['minperp_unidirectional_magnitude-'+key] = self.minperp_roughness[key]
            dfs['minperp_unidirectional_orientation-'+key] = self.minperp_roughness_dir[key]
            dfs['minperp_bidirectional_magnitude-'+key] = self.minperp_roughness_x2[key]
            dfs['minperp_bidirectional_orientation-'+key] = self.minperp_roughness_dir_x2[key]

        dfs.to_csv(*args,**kwargs)

    def print_directional_roughness(self,file_prefix:str,metric:str,**fig_kwargs):
        with PdfPages(f"{file_prefix}_{metric.replace('*','star')}.pdf") as pdf:
            print(f"Writing plots to {file_prefix}_{metric.replace('*','star')}.pdf")
            # fig,ax = plt.subplots(ncols=2,subplot_kw={"projection":"polar"},**fig_kwargs)
            fig = plt.figure()

            fig.tight_layout()
            ax = [fig.add_subplot(121,polar=True),fig.add_subplot(122)]
            self.surface._calculate_edges()
            b = self.surface.edge_bounds
            ax[1].set_aspect('equal','box')
            for data,position in tqdm(zip(self.roughness_data[metric],self.samples)):
                
                ax[0].plot(self.az,data)
                for bound in b:
                    ax[1].plot(bound[:,0],bound[:,1])
                circle = plt.Circle(position,self.sample_window.radius,fill=False,color='r')
                ax[1].add_patch(circle)
                
                pdf.savefig(fig)
                ax[0].clear()
                ax[1].clear()

    def to_vtk(self,file_prefix:str,metric:str,find_edges=False,method='linear'):
        data_path = f'{file_prefix}_roughness.vtu'
        print(f"Writing to {data_path}")
        centroids = np.mean(self.surface.original_points[self.surface.triangles],axis=1)
        normals = self.surface.original_normals 
        
        centroids = np.mean(self.surface.original_points[self.surface.triangles],axis=1)
        
        if find_edges:
            # Determine points affected by edge
            print("Drawing Boundary")
            self.surface._calculate_edges()
            bounds = self.surface.edge_bounds
            bounds = _offset_bounds(bounds,self.sample_window.radius)
            z = [griddata(self.surface.original_points[:,:2],self.surface.original_points[:,2],bound)[:,np.newaxis] for bound in bounds]
            bounds = [np.hstack([bound,z_bound]) for bound,z_bound in zip(bounds,z)]
            # mask = _centroids_in_offset_bounds(centroids,bounds,self.sample_window.radius).astype(np.uint8)
            points = np.zeros([0,3])
            cells = []
            n_cells = 0
            for bound in bounds:
                points = np.vstack([points,bound])
                cell_array = np.arange(bound.shape[0]-1)[:,np.newaxis]
                cell_array = np.hstack([cell_array,cell_array+1])
                cell_array = cell_array + n_cells
                n_cells = n_cells + cell_array.shape[0] + 1
                cells.append(("line",cell_array.astype(np.int32)))
            edge_mesh = meshio.Mesh(
                points.astype(np.float32),
                cells
            )
            edge_mesh.write(f"{file_prefix}_boundary.vtu",compression='lzma')
                
        roughness_data_vtk = {}
        print("Writing magnitude data")
        for key,val in tqdm(self.magopts.items()):
            roughness_data_vtk[key] = [griddata(self.samples,val[metric],centroids[:,:2],method=method).astype(np.float32)]
            # roughness_data_vtk[key] = roughness_data_vtk[key].tolist()
        
        
        for az, col in tqdm(zip(self.az,range(self.roughness_data[metric].shape[1]))):
            roughness_data_vtk[f"DR_{np.degrees(az):03.1f}"] = [griddata(self.samples,self.roughness_data[metric][:,col],centroids[:,:2],method=method).astype(np.float32)]
            # roughness_data_vtk[f"DR_{np.degrees(az):03.1f}"] = roughness_data_vtk[f"DR_{np.degrees(az):03.1f}"].tolist()

        for az, col in tqdm(zip(self.az[:self.roughness_data_x2[metric].shape[1]],range(self.roughness_data_x2[metric].shape[1]))):
            roughness_data_vtk[f"2DR_{np.degrees(az):03.1f}"] = [griddata(self.samples,self.roughness_data_x2[metric][:,col],centroids[:,:2],method=method).astype(np.float32)]
            # roughness_data_vtk[f"2DR_{np.degrees(az):03.1f}"] = roughness_data_vtk[f"2DR_{np.degrees(az):03.1f}"].tolist()

        # if find_edges:
        #     roughness_data_vtk['edge_mask'] = [mask]
        roughness_data_vtk['min_unidirectional_dir'] = _generate_vtkdir_data(self.min_roughness_dir[metric],self.min_roughness[metric],centroids,normals,self.samples)
        roughness_data_vtk['max_unidirectional_dir'] = _generate_vtkdir_data(self.max_roughness_dir[metric],self.max_roughness[metric],centroids,normals,self.samples)
        roughness_data_vtk['minperp_unidirectional_dir'] = _generate_vtkdir_data(self.min_roughness_dir[metric]+np.pi/2,self.min_roughness[metric],centroids,normals,self.samples)
        roughness_data_vtk['min_bidirectional_dir'] = _generate_vtkdir_data(self.min_roughness_dir_x2[metric],self.min_roughness_x2[metric],centroids,normals,self.samples)
        roughness_data_vtk['max_bidirectional_dir'] = _generate_vtkdir_data(self.max_roughness_dir_x2[metric],self.max_roughness_x2[metric],centroids,normals,self.samples)
        roughness_data_vtk['minperp_bidirectional_dir'] = _generate_vtkdir_data(self.min_roughness_dir_x2[metric]+np.pi/2,self.min_roughness_x2[metric],centroids,normals,self.samples)
        points = self.surface.original_points.astype(np.float32)
        cells = [("triangle",self.surface.triangles.astype(np.int32))]
        magnitude_mesh = meshio.Mesh(
            points,cells,
            cell_data=roughness_data_vtk,
            point_data={'elevation':self.surface.points[:,2]-np.mean(self.surface.points[:,2])})
        magnitude_mesh.write(data_path,compression='lzma')
        
        
        # print("Writing directional data")
        # dir_data = {}
        # dir_data['min_unidirectional'] = _generate_vtkdir_data(self.min_roughness_dir[metric],self.min_roughness[metric],centroids,normals,self.samples)
        # dir_data['max_unidirectional'] = _generate_vtkdir_data(self.max_roughness_dir[metric],self.max_roughness[metric],centroids,normals,self.samples)
        # dir_data['minperp_unidirectional'] = _generate_vtkdir_data(self.min_roughness_dir[metric]+np.pi/2,self.min_roughness[metric],centroids,normals,self.samples)
        # dir_data['min_bidirectional'] = _generate_vtkdir_data(self.min_roughness_dir_x2[metric],self.min_roughness_x2[metric],centroids,normals,self.samples)
        # dir_data['max_bidirectional'] = _generate_vtkdir_data(self.max_roughness_dir_x2[metric],self.max_roughness_x2[metric],centroids,normals,self.samples)
        # dir_data['minperp_bidirectional'] = _generate_vtkdir_data(self.min_roughness_dir_x2[metric]+np.pi/2,self.min_roughness_x2[metric],centroids,normals,self.samples)
        # if find_edges:
        #     dir_data['edge_mask'] = [mask]
        # dir_mesh = meshio.Mesh(
        #     centroids,[('vertex',np.arange(centroids.shape[0],dtype=np.int32)[:,np.newaxis])],
        #     cell_data=dir_data)
        # dir_mesh.write(f"{file_prefix}_directions.vtu",compression='lzma')

    def generate_spaced_streamlines(self,roughness_metric,map_type,n_lines):
        # Using RK4 for streamline
        field = self.diropts[map_type][roughness_metric]
        field = np.hstack([np.cos(field[:,np.newaxis]),np.sin(field[:,np.newaxis])])
        
        positions = self.samples

        # orient field to be continuously oriented (comb vectors)
        field = _comb_vectorfield(positions,field)
        dt = self.sample_spacing/4
        self.surface._calculate_external_edges()
        offset_bounds = _offset_bounds([self.surface.external_edge_bounds],self.sample_window.radius)[0]
        # Pick point 1
        fig,ax = plt.subplots()
        ax.quiver(
            positions[:,0],positions[:,1],
            field[:,0],field[:,1],
            headaxislength=1,headwidth=1,headlength=1
        )
        ax.plot(offset_bounds[:,0],offset_bounds[:,1])
        ax.set_aspect('equal')
        p1 = np.array(plt.ginput(timeout=-1)[0])
        ax.plot(p1[0],p1[1],'rx')
        # Pick poin t2
        p2 = np.array(plt.ginput(timeout=-1)[0])
        ax.plot(p2[0],p2[1],'rx')
        
        step = (p2 - p1)/(n_lines - 1)
        starter_positions = np.vstack([
            np.linspace(p1[0],p2[0],n_lines,endpoint=True),
            np.linspace(p1[1],p2[1],n_lines,endpoint=True)
        ]).T
        # starter_positions = np.vstack([
        #     np.arange(p1[0],p2[0]+step[0],step[0]),
        #     np.arange(p1[1],p2[1]+step[1],step[1])]).T
        samples = []
        for position in starter_positions:
            # streamline generate
            raw_positions = _streamline(
                positions, field, offset_bounds, 
                position,np.linalg.norm(self.surface.lengths),dt)
            raw_positions = np.array(raw_positions)
            
            # Resample
            samples.append(_resample_polyline(raw_positions,self.surface.resolution/4))
            
            # Plot result
            ax.plot(samples[-1][:,0],samples[-1][:,1])
        return fig,samples

    def generate_streamline_selection(self,roughness_metric:str,map_type:str):
        # Using RK4 for streamline
        bounds = self.surface.bounds()
        field = self.diropts[map_type][roughness_metric]
        field = np.hstack([np.cos(field[:,np.newaxis]),np.sin(field[:,np.newaxis])])
        
        positions = self.samples
        
        # orient field to be continuously oriented (comb vectors)
        field = _comb_vectorfield(positions,field)
        dt = self.sample_spacing/4
        self.surface._calculate_external_edges()
        offset_bounds = _offset_bounds([self.surface.external_edge_bounds],self.sample_window.radius)[0]
        # Pick point from plot
        fig,ax = plt.subplots()
        ax.quiver(
            positions[:,0],positions[:,1],
            field[:,0],field[:,1],
            # headaxislength=1,headwidth=1,headlength=1
        )
        ax.plot(offset_bounds[:,0],offset_bounds[:,1])
        ax.set_aspect('equal')
        current_point = np.array(plt.ginput(timeout=-1)[0])
        
        # streamline generate
        sample_positions = _streamline(
            positions, field, offset_bounds, 
            current_point,np.linalg.norm(self.surface.lengths),dt)
        
        # Resample line to spacing
        sample_positions = np.array(sample_positions)
        ax.plot(sample_positions[:,0],sample_positions[:,1])
        plt.show(block=False)
        print(f"Resampling polyline to {self.surface.resolution/4}")
        new_sample_positions = _resample_polyline(
            sample_positions,self.surface.resolution/2)
        
        ax.plot(new_sample_positions[:,0],new_sample_positions[:,1],'o')
        plt.show(block=False)
        return fig,new_sample_positions
        
        