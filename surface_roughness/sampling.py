from fileinput import filename
from itertools import compress
import os
from typing import Union

import numpy as np
import numexpr as ne
from pandas import DataFrame, concat
from tqdm import tqdm
from tqdm.contrib import tmap
import matplotlib.pyplot as plt
from pyevtk.hl import unstructuredGridToVTK, pointsToVTK
from pyevtk.vtk import VtkTriangle,VtkVertex

from scipy.stats import gaussian_kde
from scipy.interpolate import griddata

from .roughness_impl import (
    _rs,
    _cppDirectionalRoughness,
    _PyDirectionalRoughness,
    _PyTINBasedRoughness,
    _cppTINBasedRoughness,
    _cppTINBasedRoughness_againstshear,
    _cppTINBasedRoughness_bestfit,
    _cppMeanDipRoughness,
    DirRoughnessBase
)

from .roughness import Surface


class SampleWindow:
    def __init__(self,is_circle=True,radius=None,width=None,height=None):
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
    if sample_vertex_inclusion > 3 or sample_vertex_inclusion < 0:
        raise ValueError("Argument sample_vertex_inclusion must be in range between 0-3")
    
    return RoughnessMap(surface,roughness_method,sample_window,sample_spacing,sample_vertex_inclusion,seed_left_offset,seed_bot_offset,**roughness_kwargs)


class RoughnessMap:
    """A class providing utilities to map local roughness of a 3D surface
    """
    def __init__(self,*args,**kwargs):
        self.surface,self.roughness_method,self.sample_window,self.sample_spacing,self.sample_vertex_inclusion,self.seed_left_offset,self.seed_bot_offset = args
        self.roughness_kwargs = kwargs
        self._methods = {
            'mean_dip':_cppMeanDipRoughness,
            'delta_t':_cppTINBasedRoughness,
            'delta_n':_cppTINBasedRoughness_againstshear,
            'delta_a':_cppTINBasedRoughness_bestfit,
            'thetamax_cp1':_cppDirectionalRoughness
        }
        if not self.roughness_method in self._methods:
            raise ValueError(f"Roughness method for roughness map must be {self._methods.keys()}")
        
        self.roughness_data = {}
        self.roughness_data_x2 = {}
        self.min_roughness = {}
        self.min_roughness_dir = {}
        self.max_roughness = {}
        self.max_roughness_dir = {}
        self.min_roughness_x2 = {}
        self.min_roughness_dir_x2 = {}
        self.max_roughness_x2 = {}
        self.max_roughness_dir_x2 = {}

        self.diropts = {
            'min_unidirectional':self.min_roughness_dir,
            'max_unidirectional':self.max_roughness_dir,
            'min_bidirectional':self.min_roughness_dir_x2,
            'max_bidirectional':self.max_roughness_dir_x2
        }
        self.magopts = {
            'min_unidirectional':self.min_roughness,
            'max_unidirectional':self.max_roughness,
            'min_bidirectional':self.min_roughness_x2,
            'max_bidirectional':self.max_roughness_x2
        }
        self.label = {
            'min_bidirectional':'Min. 2DR',
            'max_bidirectional':'Max. 2DR',
            'min_unidirectional':'Min. DR',
            'max_unidirectional':'Max. DR'
        }

    @staticmethod
    def _in_circle(x,y,cx,cy,r):
        # https://stackoverflow.com/questions/59963642/is-there-a-fast-python-algorithm-to-find-all-points-in-a-dataset-which-lie-in-a
        return ne.evaluate('(x - cx)**2 + (y-cy)**2 < r**2')
    
    @staticmethod
    def _triangles_from_p_in_sample(triangles,point_in_sample):
        return np.all(point_in_sample[triangles],axis=1)


    def sample(self,verbose=False):
        """Generate samples based on the sample window and spacing provided

        :param verbose: Enables command line output, defaults to False
        :type verbose: bool, optional
        """
        if verbose:
            print("Sampling...")
        bounds = self.surface.bounds()
        n = (bounds[1,:] - bounds[0,:]) / self.sample_spacing
        n = np.ceil(n).astype(np.int)
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
        if self.sample_window.is_circle:
            if verbose:
                sample_iter = tqdm(self.samples,total=self.samples.shape[0])
            else:
                sample_iter = self.samples
            p_in_sample = np.array([RoughnessMap._in_circle(
                self.surface.points[:,0],
                self.surface.points[:,1],
                sample[0],
                sample[1],self.sample_window.radius) for sample in sample_iter])
            
            if verbose:
                p_in_sample_iter = tqdm(p_in_sample,total=len(p_in_sample))
            else:
                p_in_sample_iter = p_in_sample
            self.t_in_circle = np.array([RoughnessMap._triangles_from_p_in_sample(
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
        self.raw_roughness_calculators:list[DirRoughnessBase] = [run_calc(i,tlist) for i,tlist in tqdm(enumerate(self.t_in_circle),total=len(self.t_in_circle))]

        # collect sample properties
        self.final_orientations = np.array([c.final_orientation for c in self.raw_roughness_calculators])
        self.min_bounds = np.array([c.min_bounds for c in self.raw_roughness_calculators])
        self.max_bounds = np.array([c.max_bounds for c in self.raw_roughness_calculators])
        self.centroids = np.array([c.centroid for c in self.raw_roughness_calculators])
        self.shape_sizes = np.array([c.shape_size for c in self.raw_roughness_calculators])
        self.total_areas = np.array([c.total_area for c in self.raw_roughness_calculators])
        
        self.az = self.raw_roughness_calculators[0]['az']
        self.n_tri = [len(t_in_c) for t_in_c in self.t_in_circle]
        self.n_tri_dir = np.vstack([c['n_tri'] for c in self.raw_roughness_calculators])

    def analyze_directional_roughness(self,metric:str):
        """Process the roughness results for plotting

        :param metric: Metric to analyze directional roughness
        :type metric: str
        """
        
        self.roughness_data[metric] = np.vstack([c[metric] for c in self.raw_roughness_calculators])
        # Collect bidirectional data
        self.n_tri_dir_x2 = self.n_tri_dir[:,self.az >= np.pi] +  self.n_tri_dir[:,self.az < np.pi]
        self.roughness_data_x2[metric] = self.roughness_data[metric][:,self.az >= np.pi] +  self.roughness_data[metric][:,self.az < np.pi]
        
        # Unidirectional roughness stats
        self.min_roughness[metric] = np.amin(self.roughness_data[metric],axis=1)
        self.min_roughness_dir[metric] = self.az[np.argmin(self.roughness_data[metric],axis=1)]
        
        self.max_roughness[metric] = np.amax(self.roughness_data[metric],axis=1)
        self.max_roughness_dir[metric] = self.az[np.argmax(self.roughness_data[metric],axis=1)]

        # Bidirectional roughness stats
        self.min_roughness_x2[metric] = np.amin(self.roughness_data_x2[metric],axis=1)
        self.min_roughness_dir_x2[metric] = self.az[np.argmin(self.roughness_data_x2[metric],axis=1)]
        
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
        kern = gaussian_kde(self.magopts[stat][metric])
        x_range = np.linspace(np.amin(self.magopts[stat][metric]),np.amax(self.magopts[stat][metric]))
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
        """Plots the magnitude of requested roughness data

        :param metric: Roughness metric used for plotting
        :type metric: str
        :param stat: Statistic option from RoughnessMap.diropts.keys()
        :type stat: str
        :param colorbar_label: Colorbar label, defaults to None
        :type colorbar_label: str, optional
        :return: matplotlib.pyplot.subplots figure handle
        :rtype: Figure
        :return: matplotlib.pyplot.subplots axes handle
        :rtype: axes.Axes
        """
        if ax is None:
            fig,ax = plt.subplots(**fig_kwargs)
        maxdrmag = ax.tricontourf(self.samples[:,0],self.samples[:,1],self.magopts[stat][metric],levels=n_colours,extend='neither')
        cb = plt.colorbar(maxdrmag)
        cb.set_label(colorbar_label)

        ax.set_title(f'{self.label[stat]} magnitude')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.axis('equal')
    
    def to_csv(self,*args,**kwargs):
        def _build_df_row(calc:DirRoughnessBase):
            # flatten dataframe to row
            sample_df:DataFrame = calc.to_pandas()
            sample_df = sample_df.unstack().to_frame().sort_index(level=1).T
            sample_df.columns = [x+'_'+str(round(y*180/np.pi)) for x,y in sample_df.columns]
            
            # add properties
            for dir,val in zip(['X','Y','Z'],calc.final_orientation):
                sample_df[f'FinalOrientation_{dir}'] = val

            for dir,val in zip(['left','bot'],calc.min_bounds):
                sample_df[f'Bounds_{dir}'] = val
            for dir,val in zip(['right','top'],calc.max_bounds):
                sample_df[f'Bounds_{dir}'] = val
            for dir,val in zip(['X','Y','Z'],calc.centroid):
                sample_df[f'Centroid_{dir}'] = val
            for dir,val in zip(['X','Y','Z'],calc.shape_size):
                sample_df[f'Size_{dir}'] = val

            sample_df['Area'] = calc.total_area

            

            return sample_df

        dfs = concat([_build_df_row(c) for c in self.raw_roughness_calculators])
        for key in self.max_roughness.keys():
            dfs['MaxDR_'+key] = self.max_roughness[key]
            dfs['MaxDR_Direction'+key] = self.max_roughness_dir[key]
            dfs['Max2DR'+key] = self.max_roughness_x2[key]
            dfs['Max2DR_Direction'+key] = self.max_roughness_dir_x2[key]

            dfs['MinDR'+key] = self.min_roughness[key]
            dfs['MinDR_Direction'+key] = self.min_roughness_dir[key]
            dfs['Min2DR'+key] = self.min_roughness_x2[key]
            dfs['Min2DR_Direction'+key] = self.min_roughness_dir_x2[key]
        dfs.to_csv(*args,**kwargs)
    
    def to_vtk(self,file_prefix:str,metric:str):
        # interpolate data by triangle centroid
        centroids = np.mean(self.surface.points[self.surface.triangles,:2],axis=1)
        roughness_data_vtk = {}
        for key,val in self.magopts.items():
            roughness_data_vtk[key] = griddata(self.samples,val[metric],centroids)
        
        for az, col in zip(self.az,range(self.roughness_data[metric].shape[1])):
            roughness_data_vtk[f"DR_{np.degrees(az)}"] = griddata(self.samples,self.roughness_data[metric][:,col],centroids)

        for az, col in zip(self.az[:self.roughness_data_x2[metric].shape[1]],range(self.roughness_data_x2[metric].shape[1])):
            roughness_data_vtk[f"2DR_{np.degrees(az)}"] = griddata(self.samples,self.roughness_data_x2[metric][:,col],centroids)

        x = self.surface.points[:,0]
        y = self.surface.points[:,1]
        z = self.surface.points[:,2]

        conn = self.surface.triangles.flatten()
        offset = np.arange(3,(self.surface.triangles.shape[0]+1)*3,3).astype(np.float)

        ctype = np.ones(self.surface.triangles.shape[0]) * VtkTriangle.tid

        unstructuredGridToVTK(
            f"{file_prefix}_magnitude",x,y,z,
            connectivity=conn,offsets=offset,cell_types=ctype,cellData=roughness_data_vtk)

        # Generate direction vector file
        x = centroids[:,0]
        y = centroids[:,1]
        z = griddata(self.surface.points[:,:2],self.surface.points[:,2],centroids)
        conn = np.arange(x.shape[0])
        offset = conn.astype(np.int64)+1
        ctype = np.ones(conn.shape[0],dtype=np.int64)*VtkVertex.tid
        normals = self.surface.normals

        def generate_vtkdir_data(raw_dir_data):
            dir = raw_dir_data
            dir = np.hstack([np.cos(dir[:,np.newaxis]),np.sin(dir[:,np.newaxis])])
            vtk_dir = griddata(self.samples,dir,centroids)
            vtk_dir /= np.linalg.norm(vtk_dir,axis=1)[:,np.newaxis]
            z = -(vtk_dir[:,0]*normals[:,0] + vtk_dir[:,1]*normals[:,1])/normals[:,2]
            vtk_dir = np.hstack([vtk_dir,z[:,np.newaxis]])
            vtk_dir/=np.linalg.norm(vtk_dir,axis=1)[:,np.newaxis]
            vtk_dir = vtk_dir[:,0],vtk_dir[:,1],vtk_dir[:,2]
            return vtk_dir
        dir_data = {}
        dir_data['min_unidirectional'] = generate_vtkdir_data(self.min_roughness_dir[metric])
        dir_data['max_unidirectional'] = generate_vtkdir_data(self.max_roughness_dir[metric])
        dir_data['min_bidirectional'] = generate_vtkdir_data(self.min_roughness_dir_x2[metric])
        dir_data['max_bidirectional'] = generate_vtkdir_data(self.max_roughness_dir_x2[metric])
        
        unstructuredGridToVTK(
            f"{file_prefix}_directions.vtk",x,y,z,
            connectivity=conn.astype(np.int64),offsets=offset,cell_types=ctype,
            cellData=dir_data)